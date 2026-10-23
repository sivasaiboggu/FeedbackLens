import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class StreamlitTests(unittest.TestCase):
    def test_artifact_dashboard_and_queue_filter(self):
        before = (ROOT / "artifacts/results.json").read_bytes()
        app = AppTest.from_file(
            str(ROOT / "streamlit_app.py"), default_timeout=30
        ).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(len(app.get("plotly_chart")), 3)
        self.assertEqual(len(app.get("arrow_vega_lite_chart")), 1)
        self.assertEqual(app.metric[1].value, "79.5%")
        app.selectbox[0].select("negative").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(all(app.dataframe[0].value.predicted_sentiment == "negative"))
        self.assertEqual(before, (ROOT / "artifacts/results.json").read_bytes())
