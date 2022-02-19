import { parse } from "@vanillaes/csv";
import axios from "axios";
import { Band, GenerationInput } from "./types";

const API_URL = process.env.REACT_APP_API_URL;

function shouldSample(input: GenerationInput) {
    if (input.band_name || input.song_name) return false;
    else return true;
}

export async function submitForm(input: GenerationInput): Promise<Band> {
    if (shouldSample(input)) return (await axios.post(`${API_URL}/sample`, input)).data
    else return (await axios.post(`${API_URL}/bands`, input)).data
}

