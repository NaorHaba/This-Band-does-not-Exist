import { Typography } from "@mui/material";
import { Box } from "@mui/system";
import React from "react";
import { useQuery } from "react-query";
import { submitForm } from "./api";
import "./App.css";
import { BandForm } from "./BandForm";
import { GeneratedBand } from "./GeneratedBand";
import { Band, GenerationInput } from "./types";

function App() {
  const [triggerQuery, setTriggerQuery] = React.useState(false);

  const [bandName, setBandName] = React.useState("");
  const [genre, setGenre] = React.useState("");
  const [songName, setSongName] = React.useState("");
  const [generatedBand, setGeneratedBand] = React.useState<Band | undefined>(
    undefined
  );

  const { status, data, error, isError, remove } = useQuery(
    "submitBandForm",
    () =>
      submitForm({
        band_name: bandName,
        genre,
        song_name: songName,
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
      console.log(data);
      setGeneratedBand(data);
      remove();
    }
  }, [status, data]);

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Typography variant="h2">This Band Does Not Exist</Typography>
      <BandForm
        {...{
          bandName,
          genre,
          songName,
          status,
          setBandName,
          setGenre,
          setSongName,
          setTriggerQuery,
        }}
      />
      {generatedBand && <GeneratedBand {...generatedBand} />}
    </Box>
  );
}

export default App;
