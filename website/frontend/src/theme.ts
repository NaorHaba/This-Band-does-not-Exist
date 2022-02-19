import { createTheme, responsiveFontSizes } from "@mui/material";
import { blue, grey } from "@mui/material/colors";

export const getAppTheme = () => {
    let theme = createTheme({
        palette:
        {
            // palette values for dark mode
            primary: {
                main: '#3f9ab5',
            },
            secondary: {
                main: '#f50057',
            },
            divider: blue[200],
            text: {
                primary: "#fff",
                secondary: grey[500],
            },
            background: {
                default: "#303030",
                paper: "#424242",
            },

        },
        typography: {
            h4: {
                fontSize: '3rem',
            },
        }
    });
    theme = responsiveFontSizes(theme);
    return theme;
};