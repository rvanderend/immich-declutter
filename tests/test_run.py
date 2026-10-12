import numpy as np

import rules
from declutter import Config, Declutter
from immich import NotFound
from state import State


class FakeImmich:
    def __init__(self, assets):
        self.assets = {a["id"]: a for a in assets}
        self.album_members: dict[str, set[str]] = {}

    def ping(self):
        return {}

    def check_access(self):
        pass

    def albums(self):
        return [{"id": aid, "albumName": aid} for aid in self.album_members]

    def create_album(self, name, description):
        self.album_members[name] = set()
        return name

    def album_asset_ids(self, album_id):
        return set(self.album_members[album_id])

    def add_to_album(self, album_id, ids):
        self.album_members[album_id] |= set(ids)

    def search(self, **filters):
        return list(self.assets.values())

    def asset(self, asset_id):
        if asset_id not in self.assets:
            raise NotFound(asset_id)
        return self.assets[asset_id]

    def thumbnail(self, asset_id):
        return asset_id.encode(), "image/webp"

    def ocr(self, asset_id):
        return []


class FakeClip:
    """Thumbnail bytes = asset id; ids starting with 'meme' score as memes."""

    def load_prompts(self):
        pass

    def image(self, data, mime):
        return np.array([1.0 if data.startswith(b"meme") else 0.0])

    def scores(self, emb):
        return {"meme": float(emb[0])}


def asset(id_, make=None):
    return {"id": id_, "type": "IMAGE", "width": 1200, "height": 1600, "visibility": "timeline",
            "exifInfo": {"make": make}, "originalPath": f"/x/{id_}.jpg"}


def make(tmp_path, monkeypatch, assets):
    for k, v in {"IMMICH_URL": "http://i", "IMMICH_API_KEY": "k", "ML_URL": "http://m",
                 "FULL_SCAN": "true", "MIN_AGE_HOURS": "0", "THREADS": "1"}.items():
        monkeypatch.setenv(k, v)
    cfg = Config.from_env()
    immich = FakeImmich(assets)
    return Declutter(cfg, immich, FakeClip(), State(str(tmp_path / "s.db"))), immich, cfg


def test_full_cycle(tmp_path, monkeypatch):
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("meme1"), asset("meme2"), asset("photo"),
                                                  asset("meme3", make="Apple")])
    r = d.run()
    album = cfg.albums[rules.MEMES]
    assert r.new[rules.MEMES] == 2 and immich.album_members[album] == {"meme1", "meme2"}

    # user keeps meme1 (removes it from the album) and deletes meme2
    immich.album_members[album] = set()
    del immich.assets["meme2"]
    r = d.run()
    assert (r.kept, r.deleted, r.new[rules.MEMES]) == (1, 1, 0)
    assert immich.album_members[album] == set()  # meme1 is never proposed again


def test_albums_filled_before_state_commit(tmp_path, monkeypatch):
    """A crash after the album flush must not turn proposals into 'kept' on the next run."""
    many = [asset(f"meme{i}") for i in range(600)]
    d, immich, cfg = make(tmp_path, monkeypatch, many)
    calls = []
    orig = immich.add_to_album
    immich.add_to_album = lambda aid, ids: (calls.append(len(ids)), orig(aid, ids))
    d.run()
    assert calls == [500, 100]


def test_failed_album_flush_rolls_back_state(tmp_path, monkeypatch):
    """A failing album add must not leave 'proposed' rows behind for the next run to commit."""
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("meme1"), asset("photo")])
    real_add = immich.add_to_album

    def forbidden(aid, ids):
        raise RuntimeError("403 Forbidden")

    immich.add_to_album = forbidden
    try:
        d.run()
    except RuntimeError:
        pass
    else:
        raise AssertionError("run should fail")

    immich.add_to_album = real_add
    r = d.run()
    assert (r.kept, r.new[rules.MEMES]) == (0, 1)
    assert immich.album_members[cfg.albums[rules.MEMES]] == {"meme1"}
