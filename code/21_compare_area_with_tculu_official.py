from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
JOINT_DIR = PROJECT / "data/mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10"
TCULU_DIR = PROJECT / "data/tculu_shenzhen"
OFFICIAL_XLSX = PROJECT / "data/official_statistics/shenzhen_land_area_2000_2020.xlsx"
OUT_DIR = JOINT_DIR / "area_comparison_tculu_official"

YEARS = [2000, 2005, 2010, 2015, 2020]
CLASSES = ["production", "living", "green_space", "water", "farmland"]


def load_joint() -> pd.DataFrame:
    df = pd.read_csv(JOINT_DIR / "area_by_class_joint_all_models.csv")
    df = df[df["year"].isin(YEARS) & df["class"].isin(CLASSES)].copy()
    df["source"] = "joint_" + df["model"].astype(str)
    return df[["source", "year", "class", "area_km2"]]


def load_tculu() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TCULU_DIR / "shenzhen_tculu_area_by_class.csv")
    raw = raw[raw["year"].isin(YEARS)].copy()

    class_map = {
        "industrial": "production",
        "residential": "living",
        "commercial": "living",
        "institutional": "living",
        "green_space": "green_space",
        "water": "water",
        "farmland": "farmland",
        "bare_land": "farmland",
    }
    five = raw[raw["class_name"].isin(class_map)].copy()
    five["class"] = five["class_name"].map(class_map)
    five = (
        five.groupby(["year", "class"], as_index=False)["area_km2"]
        .sum()
        .assign(source="TCULU_5class")
    )

    built = raw.pivot_table(
        index="year", columns="class_name", values="area_km2", aggfunc="sum", fill_value=0
    ).reset_index()
    for col in [
        "industrial",
        "residential",
        "commercial",
        "institutional",
        "transport",
        "green_space",
        "water",
        "farmland",
        "bare_land",
    ]:
        if col not in built.columns:
            built[col] = 0.0
    built["tculu_production"] = built["industrial"]
    built["tculu_living"] = built["residential"] + built["commercial"] + built["institutional"]
    built["tculu_built_no_transport"] = built["tculu_production"] + built["tculu_living"]
    built["tculu_built_with_transport"] = built["tculu_built_no_transport"] + built["transport"]
    built["tculu_natural_5class"] = (
        built["green_space"] + built["water"] + built["farmland"] + built["bare_land"]
    )
    built = built[
        [
            "year",
            "tculu_production",
            "tculu_living",
            "tculu_built_no_transport",
            "tculu_built_with_transport",
            "tculu_natural_5class",
        ]
    ].copy()
    return five[["source", "year", "class", "area_km2"]], built


def load_official() -> pd.DataFrame:
    off = pd.read_excel(OFFICIAL_XLSX, header=1)
    off = off.rename(columns={"年份": "year", "指标": "indicator", "面积": "area", "单位": "unit"})
    off = off[off["year"].isin(YEARS)].copy()
    off["area_km2"] = off["area"].astype(float)
    off.loc[off["unit"].eq("公顷"), "area_km2"] = off.loc[off["unit"].eq("公顷"), "area_km2"] / 100.0

    def map_indicator(s: str) -> str | None:
        s = str(s).strip()
        if s == "城市工业建设用地面积":
            return "official_industrial"
        if s == "城市居住建设用地面积":
            return "official_residential"
        if s == "城市商业服务业设施用地面积":
            return "official_commercial"
        if s == "建成区面积":
            return "official_built_up"
        if s == "国有建设工矿仓储用地供应情况":
            return "official_industrial_land_supply"
        return None

    off["metric"] = off["indicator"].map(map_indicator)
    off = off[off["metric"].notna()].copy()
    return off[["year", "metric", "area_km2", "indicator", "unit", "来源"]]


def build_comparison() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    joint = load_joint()
    tculu_five, tculu_built = load_tculu()
    official = load_official()

    five_long = pd.concat([joint, tculu_five], ignore_index=True)
    five_wide = (
        five_long.pivot_table(index=["year", "class"], columns="source", values="area_km2", aggfunc="first")
        .reset_index()
        .sort_values(["year", "class"])
    )

    for src in ["joint_mlp", "joint_rf"]:
        if src in five_wide.columns and "TCULU_5class" in five_wide.columns:
            five_wide[f"{src}_minus_TCULU"] = five_wide[src] - five_wide["TCULU_5class"]
            five_wide[f"{src}_ratio_to_TCULU"] = five_wide[src] / five_wide["TCULU_5class"]

    joint_built = (
        joint[joint["class"].isin(["production", "living"])]
        .groupby(["source", "year"], as_index=False)["area_km2"]
        .sum()
        .pivot(index="year", columns="source", values="area_km2")
        .reset_index()
        .rename(columns={"joint_mlp": "joint_mlp_built_prod_living", "joint_rf": "joint_rf_built_prod_living"})
    )
    prod_liv = (
        five_long[five_long["class"].isin(["production", "living"])]
        .pivot_table(index=["year", "class"], columns="source", values="area_km2", aggfunc="first")
        .reset_index()
    )

    official_wide = official.pivot_table(
        index="year", columns="metric", values="area_km2", aggfunc="first"
    ).reset_index()
    built_comp = tculu_built.merge(joint_built, on="year", how="outer").merge(official_wide, on="year", how="outer")

    for col in ["joint_mlp_built_prod_living", "joint_rf_built_prod_living", "tculu_built_no_transport", "tculu_built_with_transport"]:
        if col in built_comp.columns and "official_built_up" in built_comp.columns:
            built_comp[f"{col}_minus_official_built_up"] = built_comp[col] - built_comp["official_built_up"]
            built_comp[f"{col}_ratio_to_official_built_up"] = built_comp[col] / built_comp["official_built_up"]

    class_official_comp = prod_liv.copy()
    official_for_class = official_wide[["year"] + [c for c in ["official_industrial", "official_residential", "official_commercial"] if c in official_wide.columns]]
    class_official_comp = class_official_comp.merge(official_for_class, on="year", how="left")

    five_long.to_csv(OUT_DIR / "fiveclass_area_long_joint_tculu.csv", index=False, encoding="utf-8-sig")
    five_wide.to_csv(OUT_DIR / "fiveclass_area_wide_joint_tculu.csv", index=False, encoding="utf-8-sig")
    official.to_csv(OUT_DIR / "official_area_tidy.csv", index=False, encoding="utf-8-sig")
    built_comp.to_csv(OUT_DIR / "built_area_comparison_joint_tculu_official.csv", index=False, encoding="utf-8-sig")
    class_official_comp.to_csv(OUT_DIR / "production_living_vs_official.csv", index=False, encoding="utf-8-sig")

    print("Saved outputs to:", OUT_DIR)
    print("\nFive-class comparison:")
    print(five_wide.round(2).to_string(index=False))
    print("\nBuilt-up comparison:")
    print(built_comp.round(2).to_string(index=False))
    print("\nProduction/living vs official:")
    print(class_official_comp.round(2).to_string(index=False))


if __name__ == "__main__":
    build_comparison()
