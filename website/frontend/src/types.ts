

export interface GenerationInput{
    bandName?: string,
    genre?: string,
    songName?: string
}

export interface GeneratedBand{
    bandName: string,
    genre: string,
    songName: string,
    lyrics: string
}