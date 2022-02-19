import { LoadingButton } from "@mui/lab";
import { Box, MenuItem, TextField } from "@mui/material";
import React from "react";

interface BandFormProps {
  bandName: string;
  genre: string;
  songName: string;
  status: "idle" | "error" | "loading" | "success";
  setBandName: (s: string) => void;
  setGenre: (s: string) => void;
  setSongName: (s: string) => void;
  setTriggerQuery: (b: boolean) => void;
}

export const BandForm: React.FC<BandFormProps> = (props) => {
  const {
    bandName,
    genre,
    songName,
    status,
    setBandName,
    setGenre,
    setSongName,
    setTriggerQuery,
  } = props;

  const possibleGenres = [
    "Country",
    "Electronic",
    "Folk",
    "Hip-Hop",
    "Indie",
    "Jazz",
    "Metal",
    "Pop",
    "R&B",
    "Rock",
  ];

  const genreOptions = possibleGenres.map((s) => {
    return (
      <MenuItem key={s} value={s}>
        {`${s}`}
      </MenuItem>
    );
  });
  console.log(genreOptions);

  return (
    <Box
      component={"form"}
      id={"bandForm"}
      onSubmit={(e: any) => {
        e.preventDefault();
        setTriggerQuery(true);
      }}
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        mt: 5,
      }}
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
        }}
      >
        <TextField
          error={genre === "naor"}
          label="Band Name"
          variant="outlined"
          value={bandName}
          onChange={(e) => setBandName(e.target.value)}
          helperText={genre === "naor" ? "Text field can't be naor" : null}
        />
        <TextField
          error={genre === "naor"}
          label="Genre"
          variant="outlined"
          value={genre}
          select
          onChange={(e) => setGenre(e.target.value)}
          helperText={genre === "naor" ? "Text field can't be naor" : null}
          sx={{
            minWidth: 100,
          }}
        >
          {genreOptions}
        </TextField>
        <TextField
          error={genre === "naor"}
          label="Song Name"
          variant="outlined"
          value={songName}
          onChange={(e) => setSongName(e.target.value)}
          helperText={genre === "naor" ? "Text field can't be naor" : null}
        />
      </Box>
      <LoadingButton
        variant="contained"
        type="submit"
        form={"bandForm"}
        loading={status === "loading"}
      >
        Generate Band
      </LoadingButton>
    </Box>
  );
};
