import { LoadingButton } from "@mui/lab";
import { MenuItem, TextField } from "@mui/material";
import { Box } from "@mui/system";
import React from "react";
import { useQuery } from "react-query";
import { submitForm } from "./api";
import "./App.css";
import logo from "./logo.svg";
import { GeneratedBand, GenerationInput } from "./types";

function App() {
  const [triggerQuery, setTriggerQuery] = React.useState(false);

  const [bandName, setBandName] = React.useState("");
  const [genre, setGenre] = React.useState("");
  const [songName, setSongName] = React.useState("");
  const [generatedBand, setGeneratedBand] = React.useState<
    GeneratedBand | string
  >("");

  const { status, data, error, isError, remove } = useQuery(
    "submitBandForm",
    () =>
      submitForm({
        bandName,
        genre,
        songName,
      } as GenerationInput),
    {
      enabled: triggerQuery,
      onSettled: () => {
        setTriggerQuery(false);
      },
    }
  );

  React.useEffect(() => {
    if (isError) {
      console.log(error); // TODO
    }
    if (status === "success" && data) {
      setGeneratedBand(data);
      remove();
    }
  }, [status, data]);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
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
          >
            <MenuItem key={"Rock"} value={"Rock"}>
              Rock
            </MenuItem>
            <MenuItem key={"Pop"} value={"Pop"}>
              Pop
            </MenuItem>
          </TextField>
          <TextField
            error={genre === "naor"}
            label="Song Name"
            variant="outlined"
            value={songName}
            onChange={(e) => setSongName(e.target.value)}
            helperText={genre === "naor" ? "Text field can't be naor" : null}
          />
          <LoadingButton
            variant="outlined"
            type="submit"
            form={"bandForm"}
            loading={status === "loading"}
          >
            Submit
          </LoadingButton>
        </Box>
        {generatedBand && <Box>{generatedBand}</Box>}
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
