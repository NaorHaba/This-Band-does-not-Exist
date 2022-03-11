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
import { submitForm, submitRating } from "./api";
import "./App.css";
import { BandForm, input_contains_char } from "./BandForm";
import { Footer } from "./Footer";
import { GeneratedBand } from "./GeneratedBand";
import { Band, GenerationInput, ratingObj } from "./types";

function App() {
  const writeYourOwnInstructions =
    "Make up your own band by inserting any input you like among `Band Name`, `Genre` or `Song Name`";
  const rateUsInstructions =
    "Tell us how authentic the band and its number 1 hit are in your opinion";

  const [triggerQuery, setTriggerQuery] = React.useState(true);

  const [bandName, setBandName] = React.useState("");
  const [genre, setGenre] = React.useState("");
  const [songName, setSongName] = React.useState("");
  const [generatedBand, setGeneratedBand] = React.useState<Band | undefined>(
    undefined
  );

  const [writeYourOwn, setWriteYourOwn] = React.useState(false);
  const [ratingValue, setRatingValue] = React.useState<number | null>(2);

  const [screenError, setScreenError] = React.useState("");

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
        setBandName("");
        setSongName("");
        setGenre("");
      },
    }
  );

  React.useEffect(() => {
    if (isError) {
      if (error && (error as any).response) {
        let status = (error as any).response.status;
        if (status === 500)
          setScreenError(
            "The system is overloaded at the moment, please try again in a moment."
          );
      } else {
        setScreenError(
          "The server is currently down, please contact us to know when it will be up again :)" //
        );
      }
      console.log(error);
    }
    if (status === "success" && data) {
      setGeneratedBand(data);
      setScreenError("");
      remove();
    }
    if (!input_contains_char(bandName) || !input_contains_char(songName))
      setTriggerQuery(false);
  }, [status, data, bandName, songName]);

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
        <Box sx={{ mt: 5, width: 700, mb: 4 }}>
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
            <Box>
              <Typography component="legend" align="center">
                Rate This Band
              </Typography>
              <Tooltip title={rateUsInstructions}>
                <Rating
                  name="simple-controlled"
                  value={ratingValue}
                  onChange={(event, newValue) => {
                    setRatingValue(newValue);
                    submitRating({ ratingValue: newValue } as ratingObj);
                  }}
                  sx={{
                    alignContent: "center",
                    alignItems: "center",
                  }}
                />
              </Tooltip>
            </Box>
          </Box>
          {screenError ? (
            <Typography variant="overline" align="center">
              {screenError}
            </Typography>
          ) : generatedBand ? (
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
