import GitHubIcon from "@mui/icons-material/GitHub";
import InfoIcon from "@mui/icons-material/Info";
import { AppBar, Box, Link, Typography } from "@mui/material";

export const Footer: React.FC = () => {
  return (
    <AppBar
      position="fixed"
      color="primary"
      sx={{
        top: "auto",
        justifyContent: "center",
        bottom: 0,
        minHeight: "35px",
      }}
    >
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          gap: 3,
        }}
      >
        <Typography>Pretty Message 1</Typography>
        <Box
          sx={{
            display: "flex",
            gap: 0.75,
            alignItems: "center",
          }}
        >
          <Link
            href="https://github.com/NaorHaba/This-Band-does-not-Exist"
            underline="hover"
            target="_blank"
            rel="noopener"
            color="white"
          >
            <Typography>Github </Typography>
          </Link>
          <GitHubIcon sx={{ height: "18px", width: "18px" }} />
        </Box>
        <Box
          sx={{
            display: "flex",
            gap: 0.75,
            alignItems: "center",
          }}
        >
          <Link
            href="https://docs.google.com/forms/d/e/1FAIpQLSe7GbkAkIdTgJ3QkGmJMHhkIpjWz_I0ZX608FlxVLeT0cyJJQ/viewform?usp=sf_link"
            underline="hover"
            target="_blank"
            rel="noopener"
            color="white"
          >
            <Typography>Pretty Link 1</Typography>
          </Link>
          <InfoIcon sx={{ height: "18px", width: "18px" }} />
        </Box>
      </Box>
    </AppBar>
  );
};
