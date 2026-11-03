"""Real Estate Investment Advisor: Streamlit front end (Step 9).

Run from the project root:
    streamlit run app/app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from inference import (
    AMENITIES, GOOD_GROWTH_THRESHOLD, HORIZON_YEARS, TRANSPORT_MAP,
    build_features, city_context, fmt_lakhs, load_models, load_reference,
    predict, price_what_if, projection_path,
)

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

GREEN, RED, BLUE, GREY = "#2e9e5b", "#d9534f", "#3b7ddd", "#8a8f98"


# --- Cached loaders ----------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return load_models()


@st.cache_data(show_spinner="Loading market data…")
def get_reference():
    return load_reference()


try:
    clf, reg, feature_cols = get_models()
    ref = get_reference()
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a required file: `{e.filename}`. Run the notebook through "
        "Step 8 first so `models/` and `data/cleaned_housing.csv` exist, and start "
        "the app from the project root with `streamlit run app/app.py`."
    )
    st.stop()

opts, rng = ref["options"], ref["ranges"]


def num_kwargs(col, step):
    """min / max / default for a number widget, taken from the training data."""
    lo, hi, med = rng[col]
    cast = int if isinstance(step, int) else float
    return dict(min_value=cast(lo), max_value=cast(hi), value=cast(round(med)), step=step)


# --- Header ------------------------------------------------------------------
st.title("🏠 Real Estate Investment Advisor")
st.caption(
    f"Describe a property and get a verdict on whether it's a good investment "
    f"(≥ {GOOD_GROWTH_THRESHOLD:.0%} projected annual growth) plus an estimated "
    f"price {HORIZON_YEARS} years from now."
)

# --- Input form --------------------------------------------------------------
with st.form("property_form"):
    st.subheader("Property")
    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", opts["City"])
        property_type = st.selectbox("Property type", opts["Property_Type"])
        bhk = st.number_input("BHK", **num_kwargs("BHK", 1))
        size = st.number_input("Size (sq ft)", **num_kwargs("Size_in_SqFt", 50))
    with c2:
        price = st.number_input("Asking price (₹ lakhs)", **num_kwargs("Price_in_Lakhs", 5.0))
        year_built = st.number_input("Year built", **num_kwargs("Year_Built", 1))
        total_floors = st.number_input("Total floors in building", **num_kwargs("Total_Floors", 1))
        floor_no = st.number_input("Floor number", **num_kwargs("Floor_No", 1))
    with c3:
        furnished = st.selectbox("Furnishing", opts["Furnished_Status"])
        facing = st.selectbox("Facing", opts["Facing"])
        owner_type = st.selectbox("Owner type", opts["Owner_Type"])
        availability = st.selectbox("Availability", opts["Availability_Status"])

    st.subheader("Neighbourhood & amenities")
    d1, d2, d3 = st.columns(3)
    with d1:
        schools = st.slider("Nearby schools", **num_kwargs("Nearby_Schools", 1))
        hospitals = st.slider("Nearby hospitals", **num_kwargs("Nearby_Hospitals", 1))
    with d2:
        transport = st.select_slider("Public transport access", options=list(TRANSPORT_MAP), value="Medium")
        amenities = st.multiselect("Amenities", AMENITIES)
    with d3:
        parking = st.checkbox("Parking space")
        security = st.checkbox("Security")

    submitted = st.form_submit_button("Evaluate investment", type="primary")

if not submitted:
    st.info("Fill in the details above and press **Evaluate investment**.")
    st.stop()

if floor_no > total_floors:
    st.error("Floor number can't be higher than the total floors in the building.")
    st.stop()

# --- Predict -----------------------------------------------------------------
X = build_features(
    {
        "City": city, "Property_Type": property_type, "BHK": bhk, "Size_in_SqFt": size,
        "Price_in_Lakhs": price, "Year_Built": year_built, "Floor_No": floor_no,
        "Total_Floors": total_floors, "Furnished_Status": furnished, "Facing": facing,
        "Owner_Type": owner_type, "Availability_Status": availability,
        "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking, "Security": security, "Amenities": amenities,
    },
    feature_cols,
)
res = predict(clf, reg, X)
ctx = city_context(ref, city, price, size)

# --- Headline results --------------------------------------------------------
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Good investment?", "Yes ✅" if res.is_good else "No ❌")
m2.metric("Confidence", f"{res.confidence:.0%}")
m3.metric(
    f"Estimated price in {HORIZON_YEARS} years",
    fmt_lakhs(res.price_5y),
    delta=f"{res.total_gain:+.0%}  (≈ {res.cagr:.1%} / yr)",
)

direction = "below" if ctx["vs_median"] < 0 else "above"
summary = (
    f"The model gives this property a **{res.p_good:.0%}** chance of growing at "
    f"{GOOD_GROWTH_THRESHOLD:.0%}+ per year. At ₹{ctx['ppsf']:,.0f} per sq ft it is "
    f"**{abs(ctx['vs_median']):.0%} {direction}** the {city} median."
)
(st.success if res.is_good else st.warning)(summary)

# --- Charts ------------------------------------------------------------------
LAYOUT = dict(height=340, margin=dict(l=20, r=20, t=50, b=40))
col_a, col_b = st.columns(2)

# 1. Probability gauge
with col_a:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=res.p_good * 100,
        number={"suffix": "%"},
        title={"text": "Probability of a good investment"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": GREEN if res.is_good else RED},
            "steps": [{"range": [0, 50], "color": "rgba(217,83,79,0.15)"},
                      {"range": [50, 100], "color": "rgba(46,158,91,0.15)"}],
            "threshold": {"line": {"color": GREY, "width": 3}, "value": 50},
        },
    ))
    gauge.update_layout(**LAYOUT)
    st.plotly_chart(gauge)

# 2. Price projection
with col_b:
    path = projection_path(res.price_now, res.cagr)
    proj = go.Figure(go.Scatter(
        x=path["Year"], y=path["Price_in_Lakhs"], mode="lines+markers+text",
        text=[fmt_lakhs(v) if i in (0, len(path) - 1) else "" for i, v in enumerate(path["Price_in_Lakhs"])],
        textposition="top center", line=dict(color=BLUE, width=3),
        hovertemplate="Year %{x}: %{y:,.1f} lakhs<extra></extra>",
    ))
    proj.update_layout(
        title=f"Estimated value over {HORIZON_YEARS} years (constant annual growth)",
        xaxis_title="Years from now", yaxis_title="Price (₹ lakhs)", **LAYOUT,
    )
    st.plotly_chart(proj)

col_c, col_d = st.columns(2)

# 3. Price per sq ft vs the city
with col_c:
    vals = ctx["values"]
    lo = min(np.percentile(vals, 1), ctx["ppsf"]) * 0.95
    hi = max(np.percentile(vals, 99), ctx["ppsf"]) * 1.05
    hist = go.Figure(go.Histogram(x=vals, nbinsx=40, marker_color="rgba(59,125,221,0.55)",
                                  name=f"{city} listings"))
    hist.add_vline(x=ctx["median"], line_dash="dash", line_color=GREY,
                   annotation_text="City median", annotation_position="top left")
    hist.add_vline(x=ctx["ppsf"], line_color=GREEN if res.is_good else RED, line_width=3,
                   annotation_text="This property", annotation_position="top right")
    hist.update_layout(
        title=f"Price per sq ft vs {city} ({ctx['percentile']:.0f}th percentile)",
        xaxis_title="₹ per sq ft", yaxis_title="Listings", xaxis_range=[lo, hi],
        showlegend=False, **LAYOUT,
    )
    st.plotly_chart(hist)

# 4. What-if: asking price
with col_d:
    wi = price_what_if(clf, X)
    wif = go.Figure()
    wif.add_trace(go.Scatter(x=wi["Price_in_Lakhs"], y=wi["p_good"] * 100, mode="lines",
                             line=dict(color=BLUE, width=3), name="P(good)",
                             hovertemplate="%{x:,.1f} lakhs: %{y:.0f}%<extra></extra>"))
    wif.add_trace(go.Scatter(x=[res.price_now], y=[res.p_good * 100], mode="markers",
                             marker=dict(size=13, color=GREEN if res.is_good else RED),
                             name="Your price"))
    wif.add_hline(y=50, line_dash="dash", line_color=GREY)
    wif.update_layout(
        title="What if the asking price changed?",
        xaxis_title="Asking price (₹ lakhs)", yaxis_title="Probability of good investment (%)",
        yaxis_range=[0, 100], showlegend=False, **LAYOUT,
    )
    st.plotly_chart(wif)

good_prices = wi.loc[wi["p_good"] >= 0.5, "Price_in_Lakhs"]
if good_prices.empty:
    st.caption("No asking price between 60% and 140% of yours makes this a good investment; "
               "other features (location, amenities, infrastructure) are the limiting factor.")
elif res.is_good:
    st.caption(f"With everything else unchanged, the model still rates it a good investment "
               f"up to about **{fmt_lakhs(good_prices.max())}**.")
else:
    st.caption(f"With everything else unchanged, the model would rate it a good investment "
               f"at around **{fmt_lakhs(good_prices.max())}** or lower.")

# --- Transparency ------------------------------------------------------------
with st.expander("Features sent to the model"):
    st.dataframe(X.T.astype(str).rename(columns={0: "value"}))

with st.expander("How this works & limitations"):
    st.markdown(
        f"""
- **Verdict & confidence:** an XGBoost classifier (test accuracy 0.86, F1 0.83, ROC-AUC 0.94).
  Confidence is the probability of whichever answer it gave.
- **5-year price:** an XGBoost regressor (test R² 0.997, MAE ≈ ₹7.4 lakhs).
- **What "good investment" means:** in this dataset it is a *rule-based label*:
  projected growth ≥ {GOOD_GROWTH_THRESHOLD:.0%} / year, where growth is built from
  infrastructure, amenities and whether the price/sq ft is under the city median.
  The models learned that rule, not real market returns, so treat the output as an
  illustration of the project, not financial advice.
"""
    )