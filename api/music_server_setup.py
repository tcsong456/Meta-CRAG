import os
import pickle
from utils import logger
from .fast_bm25 import BM25

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "mock_api/cragkg")
class MusicKG(object):
    def __init__(self):
        artist_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_dict_simplified.pickle")
        logger.info(f"Reading artist dictionary from: {artist_dict_path}")
        with open(artist_dict_path, 'rb') as file:
            self.artist_dict = pickle.load(file)

        song_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_simplified.pickle")
        logger.info(f"Reading song dictionary from: {song_dict_path}")
        with open(song_dict_path, 'rb') as file:
            self.song_dict = pickle.load(file)

        grammy_df_path = os.path.join(KG_BASE_DIRECTORY, "music", "grammy_df.csv")
        logger.info(f"Reading Grammy DataFrame from: {grammy_df_path}")
        self.grammy_df = pd.read_csv(grammy_df_path)

        rank_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "rank_dict_hot100.pickle")
        logger.info(f"Reading rank dictionary for Hot 100 from: {rank_dict_hot_path}")
        with open(rank_dict_hot_path, 'rb') as file:
            self.rank_dict_hot = pickle.load(file)

        song_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_hot100.pickle")
        logger.info(f"Reading song dictionary for Hot 100 from: {song_dict_hot_path}")
        with open(song_dict_hot_path, 'rb') as file:
            self.song_dict_hot = pickle.load(file)

        artist_work_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_work_dict.pickle")
        logger.info(f"Reading artist work dictionary from: {artist_work_dict_path}")
        with open(artist_work_dict_path, 'rb') as file:
            self.artist_work_dict = pickle.load(file)
        
        self.key_map_artist = {}
        self.corpus_artist = []
        for e in self.artist_dict.keys():
            ne = self.normalize(e)
            ne_split = str(ne.split())
            if ne_split not in self.key_map_artist:
                self.key_map_artist[ne_split] = []
            self.key_map_artist[ne_split].append(e)
            self.corpus_artist.append(ne)
        self.corpus_artist = list(set(self.corpus_artist))
        self.corpus_artist.sort()
        self.corpus_artist = [ne.split() for ne in self.corpus_artist]
        self.bm25_artist = BM25(self.corpus_artist)

        self.key_map_song = {}
        self.corpus_song = []
        for e in self.song_dict.keys():
            ne = self.normalize(e)
            ne_split = str(ne.split())
            if ne_split not in self.key_map_song:
                self.key_map_song[ne_split] = []
            self.key_map_song[ne_split].append(e)
            self.corpus_song.append(ne)
        self.corpus_song = list(set(self.corpus_song))
        self.corpus_song.sort()
        self.corpus_song = [ne.split() for ne in self.corpus_song]
        self.bm25_song = BM25(self.corpus_song)

#%%
from itertools import islice
for k, v in islice(artist_dict.items(), 10):
    print(k, v)
    break

#%%
import pandas as pd
# artist_dict.to_csv('mock_api/cragkg/music/grammy_df.csv', index=False)
print(pd.__version__)
# d = pd.read_csv('mock_api/cragkg/music/grammy_df.csv')
