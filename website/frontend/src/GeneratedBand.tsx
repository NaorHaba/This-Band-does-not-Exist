import { Box, Paper, Typography } from "@mui/material";
import { Band } from "./types";

export const GeneratedBand: React.FC<Band> = (generatedBand) => {
  const lyrics = generatedBand.lyrics
    .split("\n")
    .map((lyricsRow) => <Typography>{`${lyricsRow}`}</Typography>);
  return (
    <Paper
      sx={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        width: 700,
        mt: 3,
      }}
    >
      <Typography variant="h4">{`${generatedBand.band_name}`}</Typography>
      <Typography variant="subtitle2">{`Genre: ${generatedBand.genre}`}</Typography>
      <Typography
        variant="h5"
        sx={{ mt: 3 }}
      >{`Number 1 Hit Song:`}</Typography>
      <Typography variant="h6">{`"${generatedBand.song_name}"`}</Typography>
      <Typography sx={{ mt: 3 }}></Typography>
      <>{lyrics}</>
    </Paper>
  );
};
