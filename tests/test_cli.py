from metriclens.cli import main


def test_cli_writes_report(tmp_path, messy_df, capsys):
    csv = tmp_path / "d.csv"
    messy_df.to_csv(csv, index=False)
    out = tmp_path / "r.html"
    js = tmp_path / "r.json"
    rc = main(["report", str(csv), "--target", "y", "--out", str(out), "--json", str(js),
               "--answer", "metric_priority=symmetric", "--plotlyjs", "none"])
    assert rc == 0 and out.exists() and js.exists()
    assert "METRICLENS REPORT" in capsys.readouterr().out
