

export interface GenerationInput{
    band_name?: string,
    genre?: string,
    song_name?: string
}

export interface GeneratedBand{
    band_name: string,
    genre: string,
    song_name: string,
    lyrics: string
}