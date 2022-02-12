import axios from "axios";
import { GeneratedBand, GenerationInput } from "./types";

const API_URL = process.env.REACT_APP_API_URL;

export async function submitForm(input: GenerationInput): Promise<GeneratedBand> {
    return (await axios.post(`${API_URL}/bands`, input)).data
}