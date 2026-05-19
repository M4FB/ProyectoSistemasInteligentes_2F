from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Input, Static, RadioSet, RadioButton, Select, Label, Rule
from textual.containers import Horizontal, Container
from textual.validation import Validator, ValidationResult
from textual import on

from engine import setup, predict, FEATURE_COLS

M2_TO_SQFT = 10.7639


class IntRange(Validator):
    def __init__(self, lo: int, hi: int) -> None:
        super().__init__()
        self._lo, self._hi = lo, hi

    def validate(self, value: str) -> ValidationResult:
        try:
            n = int(value)
        except ValueError:
            return self.failure("Debe ser un entero")
        if not self._lo <= n <= self._hi:
            return self.failure(f"{self._lo} – {self._hi}")
        return self.success()


class HousePriceApp(App):

    _LIMITS: dict[str, tuple[int, int, str]] = {
        "area":      (5,    2_000,  "Area (m²)"),
        "yearbuilt": (1800, 2024,   "Año construido"),
        "bedrooms":  (1,    20,     "Habitaciones"),
        "bathrooms": (1,    10,     "Baños"),
        "floors":    (1,    5,      "Pisos"),
    }

    CSS = """
    Screen {
        background: $surface;
    }
    #titulo {
        text-align: center;
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    #nota-svr {
        text-align: center;
        color: $success;
        padding: 0 0 1 0;
    }
    RadioSet {
        margin: 0 2 1 2;
    }
    .fila {
        height: 3;
        margin: 0 2;
    }
    .etiqueta {
        width: 22;
        content-align: right middle;
        padding-right: 1;
        color: $text-muted;
    }
    Input  { width: 24; }
    Select { width: 24; }
    .btn-row {
        align: center middle;
        height: 3;
        margin: 1 0;
    }
    #resultado {
        margin: 1 2;
        padding: 1 2;
        border: solid $accent;
        height: 7;
    }
    Input.-invalid {
        border: tall $error;
    }
    """

    def __init__(self, modelos, maes):
        super().__init__()
        self.modelos       = modelos
        self.maes          = maes
        self.modelo_actual = "SVR"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("Predictor de Precio de Casas", id="titulo")
        yield Static("★  SVR es el modelo más preciso de los tres", id="nota-svr")
        yield Rule()

        yield Label("  Selecciona el modelo:", classes="etiqueta")
        with RadioSet(id="selector-modelo"):
            yield RadioButton("KNN")
            yield RadioButton("SVR  ★", value=True)
            yield RadioButton("Árbol de Decisión")

        yield Rule()

        with Horizontal(classes="fila"):
            yield Label("Area (m²):",       classes="etiqueta")
            yield Input(value="167", id="area",       placeholder="5 – 2 000",
                        validators=[IntRange(5, 2_000)])
            yield Label("  Año construido:", classes="etiqueta")
            yield Input(value="1990", id="yearbuilt", placeholder="1800 – 2024",
                        validators=[IntRange(1800, 2024)])

        with Horizontal(classes="fila"):
            yield Label("Habitaciones:",    classes="etiqueta")
            yield Input(value="3",  id="bedrooms",  placeholder="1 – 20",
                        validators=[IntRange(1, 20)])
            yield Label("  Ubicación:",     classes="etiqueta")
            yield Select(
                [("0 – Rural", 0), ("1 – Suburban", 1), ("2 – Urban", 2), ("3 – Downtown", 3)],
                id="location", value=2,
            )

        with Horizontal(classes="fila"):
            yield Label("Baños:",           classes="etiqueta")
            yield Input(value="2",  id="bathrooms", placeholder="1 – 10",
                        validators=[IntRange(1, 10)])
            yield Label("  Condición:",     classes="etiqueta")
            yield Select(
                [("0 – Poor", 0), ("1 – Fair", 1), ("2 – Good", 2),
                 ("3 – Very Good", 3), ("4 – Excellent", 4)],
                id="condition", value=2,
            )

        with Horizontal(classes="fila"):
            yield Label("Pisos:",           classes="etiqueta")
            yield Input(value="1",  id="floors",    placeholder="1 – 5",
                        validators=[IntRange(1, 5)])
            yield Label("  Garage:",        classes="etiqueta")
            yield Select([("No", 0), ("Sí", 1)], id="garage", value=0)

        with Horizontal(classes="fila"):
            yield Label("Vista (0 – 4):",   classes="etiqueta")
            yield Select([(str(i), i) for i in range(5)], id="view", value=0)

        yield Rule()
        with Horizontal(classes="btn-row"):
            yield Button("  Calcular Precio  ", id="btn-calcular", variant="primary")

        with Container(id="resultado"):
            yield Static("Ingresa los datos y presiona Calcular.", id="resultado-texto")

        yield Footer()

    @on(RadioSet.Changed, "#selector-modelo")
    def cambiar_modelo(self, event: RadioSet.Changed) -> None:
        self.modelo_actual = ["KNN", "SVR", "Árbol"][event.index]

    @on(Button.Pressed, "#btn-calcular")
    def calcular(self, _) -> None:
        errores = []
        int_vals: dict[str, int] = {}

        for field, (lo, hi, nombre) in self._LIMITS.items():
            raw = self.query_one(f"#{field}", Input).value.strip()
            try:
                n = int(raw)
            except ValueError:
                errores.append(f"• {nombre}: debe ser un entero")
                continue
            if not lo <= n <= hi:
                errores.append(f"• {nombre}: debe estar entre {lo} y {hi}")
                continue
            int_vals[field] = n

        if errores:
            self.query_one("#resultado-texto", Static).update(
                "⚠  Corrige los siguientes campos:\n" + "\n".join(errores))
            return

        int_vals["area"] = round(int_vals["area"] * M2_TO_SQFT)

        features = {
            **int_vals,
            "location":  self.query_one("#location",  Select).value,
            "condition": self.query_one("#condition", Select).value,
            "garage":    self.query_one("#garage",    Select).value,
            "view":      self.query_one("#view",      Select).value,
        }

        precio = predict(self.modelos[self.modelo_actual], features)
        mae    = self.maes[self.modelo_actual]
        self.query_one("#resultado-texto", Static).update(
            f"Modelo:            {self.modelo_actual}\n"
            f"Precio estimado:   ${precio:,.0f}\n"
            f"Rango aproximado:  ${max(0, precio - mae):,.0f}  –  ${precio + mae:,.0f}\n"
            f"Margen de error:   ± ${mae:,.0f}  (MAE del modelo)"
        )


if __name__ == "__main__":
    modelos, maes = setup()
    HousePriceApp(modelos, maes).run()
