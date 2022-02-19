import {
  Button,
  CircularProgress,
  Rating,
  Skeleton,
  Tooltip,
  Typography,
} from "@mui/material";
import { Box } from "@mui/system";
import React from "react";
import { useQuery } from "react-query";
import { submitForm } from "./api";
import "./App.css";
import { BandForm } from "./BandForm";
import { Footer } from "./Footer";
import { GeneratedBand } from "./GeneratedBand";
import { Band, GenerationInput } from "./types";

function App() {
  const writeYourOwnInstructions =
    "Make up your own band by inserting any input you like among `Band Name`, `Genre` or `Song Name`";

  const [triggerQuery, setTriggerQuery] = React.useState(true);

  const [bandName, setBandName] = React.useState("");
  const [genre, setGenre] = React.useState("");
  const [songName, setSongName] = React.useState("");
  const [generatedBand, setGeneratedBand] = React.useState<Band | undefined>(
    undefined
  );

  const [writeYourOwn, setWriteYourOwn] = React.useState(false);
  const [ratingValue, setRatingValue] = React.useState<number | null>(2);

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
        setWriteYourOwn(false);
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
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Typography variant="h2" sx={{ mt: 5 }}>
        This Band Does Not Exist
      </Typography>
      {writeYourOwn ? (
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
            setWriteYourOwn,
          }}
        />
      ) : (
        <Box sx={{ mt: 5, width: 700 }}>
          <Box
            sx={{
              display: "flex",
              flexDirection: "row",
              justifyContent: "space-between",
            }}
          >
            <Box
              sx={{
                display: "flex",
                flexDirection: "row",
                alignContent: "justify",
              }}
            >
              <Button
                variant="text"
                onClick={() => {
                  setTriggerQuery(true);
                }}
              >
                New Band
              </Button>
              <Tooltip title={writeYourOwnInstructions}>
                <Button
                  variant="text"
                  onClick={() => {
                    setWriteYourOwn(true);
                  }}
                >
                  Write Your Own
                </Button>
              </Tooltip>
            </Box>
            {/* <Typography component="legend">Rate Our Generation:</Typography> */}
            <Box>
              <Typography component="legend" align="center">
                Rate This
              </Typography>
              <Rating
                name="simple-controlled"
                value={ratingValue}
                onChange={(event, newValue) => {
                  setRatingValue(newValue); // TODO
                }}
                sx={{
                  alignContent: "center",
                  alignItems: "center",
                }}
              />
            </Box>
          </Box>
          {generatedBand ? (
            <GeneratedBand {...generatedBand} />
          ) : (
            <CircularProgress />
          )}
        </Box>
      )}
      <Footer />
    </Box>
  );
}

export default App;
