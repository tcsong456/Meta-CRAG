from api.local_server import CRAG

class MusicTool:
    def __init__(self):
        self.api = CRAG()
    
    def search_artist_by_name(self, query):
        return self.api.music_search_artist_name(query)['result']
    
    def search_song_by_name(self, query):
        return self.api.music_search_song_name(query)['result']
    
    def get_artist_name(self, query):
        names = self.search_artist_by_name(query)
        for name in names:
            if name.lower() == query.lower():
                return name
        return None
    
    def get_song_name(self, query):
        names = self.search_song_by_name(query)
        for name in names:
            if name.lower() == query.lower():
                return name
        return None
    
    def get_billboard_rank(self, rank, date):
        return self.api.music_billboard_rank(rank, date)['result']
    
    def get_billboard_attributes(self, date, attribute, song_name):
        return self.api.music_billboard_attribute(date, attribute, song_name)['result']
    
    def get_grammy_best_artist_by_year(self, year):
        return self.api.music_search_grammy_artist(year)['result']
    
    def get_grammy_artist_award_count(self, artist_name):
        return self.api.music_search_grammy_artist_award_count(artist_name)['result']
    
    def get_grammy_song_award_count(self, song_name):
        return self.api.music_search_grammy_song_award_count(song_name)['result']
    
    def get_grammy_best_song_by_year(self, year):
        return self.api.music_search_best_song(year)['result']
    
    def get_grammy_artist_award(self, artist_name):
        return self.api.music_search_artist_award(artist_name)['result']
    
    def get_grammy_best_album_by_year(self, year):
        return self.api.music_best_album(year)['result']
    
    def get_grammy_artist_all_awards(self):
        return self.api.music_artist_all_awards()['result']
    
    def get_artist_birth_place(self, artist_name):
        return self.api.music_search_artist_birth_place(artist_name)['result']
    
    def get_artist_birth_date(self, artist_name):
        return self.api.music_search_artist_birth_date(artist_name)['result']
    
    def get_band_members(self, band_name):
        return self.api.music_get_band_members(band_name)['result']
    
    def get_lifespan(self, artist_name):
        return self.api.music_search_life_span(artist_name)['result']
    
    def get_song_author(self, song_name):
        return self.api.music_search_author(song_name)['result']
    
    def get_song_realse_country(self, song_name):
        return self.api.music_search_song_country(song_name)['result']
    
    def get_sonng_release_date(self, song_name):
        return self.api.music_song_release_date(song_name)['result']
    
    def get_artist_all_works(self, artist_name):
        return self.api.music_search_artist_works(artist_name)['result']