import { LoadingButton } from "@mui/lab";
import { Box, Button, MenuItem, TextField } from "@mui/material";
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
  setWriteYourOwn: (b: boolean) => void;
}

export function input_contains_char(input: string) {
  const regex = /[a-zA-Z]/;
  return input.length == 0 || regex.test(input);
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
    setWriteYourOwn,
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

  return (
    <Box
      component={"form"}
      id={"bandForm"}
      onSubmit={(e: any) => {
        e.preventDefault();
        if (input_contains_char(bandName) && input_contains_char(songName))
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
          gap: 1,
        }}
      >
        <TextField
          error={!input_contains_char(bandName)}
          label="Band Name"
          variant="outlined"
          value={bandName}
          onChange={(e) => setBandName(e.target.value)}
          helperText={
            !input_contains_char(bandName)
              ? "Must contain at least 1 alphabet character"
              : null
          }
          sx={{
            minWidth: 270,
          }}
        />
        <TextField
          label="Genre"
          variant="outlined"
          value={genre}
          select
          onChange={(e) => setGenre(e.target.value)}
          sx={{
            minWidth: 100,
          }}
        >
          {genreOptions}
        </TextField>
        <TextField
          error={!input_contains_char(songName)}
          label="Song Name"
          variant="outlined"
          value={songName}
          onChange={(e) => setSongName(e.target.value)}
          helperText={
            !input_contains_char(songName)
              ? "Must contain at least 1 alphabet character"
              : null
          }
          sx={{
            minWidth: 270,
          }}
        />
      </Box>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          gap: 1,
          alignItems: "center",
        }}
      >
        <LoadingButton
          variant="contained"
          type="submit"
          form={"bandForm"}
          loading={status === "loading"}
        >
          Generate Band
        </LoadingButton>
        <Button
          variant="contained"
          type="submit"
          onClick={(e) => {
            e.preventDefault();
            setWriteYourOwn(false);
            setTriggerQuery(false);
            setBandName("");
            setSongName("");
            setGenre("");
          }}
          form={"bandForm"}
        >
          Cancel
        </Button>
      </Box>
    </Box>
  );
};
