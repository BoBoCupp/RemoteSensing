"""
Science Olympiad 2026 Remote Sensing - Energy Balance & Physics
================================================================
Section 3c (25%): Remote sensing physics
Section 3f (20%): Energy balance modeling and climate change

Practice problems and visualizations for:
  - Blackbody radiation (Wien's Law, Stefan-Boltzmann, Rayleigh-Jeans)
  - Orbital mechanics (Kepler's Laws, orbit types)
  - Imaging fidelity (GSD, IFOV, FOV, revisit)
  - Planetary equilibrium temperature
  - Albedo and radiative forcing
  - Climate feedback modeling

No Earth Engine needed - pure physics calculations.
Generates study diagrams saved to ./output/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Physical constants
SIGMA = 5.670374419e-8   # Stefan-Boltzmann constant (W/m²/K⁴)
WIEN_B = 2.8977719e-3    # Wien's displacement constant (m·K)
K_B = 1.380649e-23       # Boltzmann constant (J/K)
C = 2.998e8              # Speed of light (m/s)
H = 6.626e-34            # Planck constant (J·s)
R_SUN = 6.96e8           # Solar radius (m)
T_SUN = 5778             # Solar temperature (K)
R_EARTH = 6.371e6        # Earth radius (m)
D_EARTH_SUN = 1.496e11   # Earth-Sun distance (m)
G = 6.674e-11            # Gravitational constant
M_EARTH = 5.972e24       # Earth mass (kg)

def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")




# ================================================================
# 1. BLACKBODY RADIATION - Planck curves for Sun, Earth, fire
# ================================================================
print("[1/7] Blackbody Radiation Curves...")

def planck(wavelength_m, T):
    """Planck's law: spectral radiance B(λ,T) in W/sr/m³."""
    a = 2 * H * C**2 / wavelength_m**5
    b = np.exp(H * C / (wavelength_m * K_B * T)) - 1
    return a / b

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Solar spectrum
wl_sun = np.linspace(0.1e-6, 4e-6, 1000)  # 0.1 to 4 micrometers
for T, label, color in [
    (5778, "Sun (5778 K)", "orange"),
    (4000, "Cooler star (4000 K)", "red"),
    (7000, "Hotter star (7000 K)", "blue"),
]:
    peak = WIEN_B / T * 1e6  # peak in micrometers
    radiance = planck(wl_sun, T)
    axes[0].plot(wl_sun * 1e6, radiance / 1e12, color=color, linewidth=2,
                 label=f"{label}, λ_peak={peak:.2f}μm")

# Mark visible spectrum
axes[0].axvspan(0.38, 0.7, alpha=0.15, color="green", label="Visible light")
axes[0].set_title("Blackbody Curves - Stars (Wien's Law)", fontsize=13)
axes[0].set_xlabel("Wavelength (μm)")
axes[0].set_ylabel("Spectral Radiance (TW/sr/m³)")
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, 3)
axes[0].grid(True, alpha=0.3)

# Earth/atmosphere spectrum
wl_earth = np.linspace(3e-6, 50e-6, 1000)  # 3 to 50 micrometers
for T, label, color in [
    (288, "Earth surface (288 K, 15°C)", "red"),
    (255, "Effective emission (255 K)", "blue"),
    (220, "Cloud tops (220 K)", "cyan"),
]:
    peak = WIEN_B / T * 1e6
    radiance = planck(wl_earth, T)
    axes[1].plot(wl_earth * 1e6, radiance / 1e6, color=color, linewidth=2,
                 label=f"{label}, λ_peak={peak:.1f}μm")

axes[1].set_title("Blackbody Curves - Earth (Thermal IR)", fontsize=13)
axes[1].set_xlabel("Wavelength (μm)")
axes[1].set_ylabel("Spectral Radiance (MW/sr/m³)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Wien's Law: λ_peak = b/T where b = 2898 μm·K\n"
             "Stefan-Boltzmann: Total power = σT⁴", fontsize=12, y=1.04)
fig.tight_layout()
save_figure(fig, "15_blackbody_radiation.png")

# Print key values
print("\n  === Key Blackbody Values (memorize these!) ===")
print(f"  Sun peak wavelength:   {WIEN_B/T_SUN*1e6:.2f} μm (visible)")
print(f"  Earth peak wavelength: {WIEN_B/288*1e6:.1f} μm (thermal IR)")
print(f"  Sun power/area:        {SIGMA * T_SUN**4 / 1e6:.1f} MW/m²")
print(f"  Earth emission/area:   {SIGMA * 288**4:.1f} W/m²")
print(f"  Solar constant at Earth: ~1361 W/m²")


# ================================================================
# 2. ELECTROMAGNETIC SPECTRUM - Remote Sensing Bands
# ================================================================
print("\n[2/7] Electromagnetic Spectrum for Remote Sensing...")

fig, ax = plt.subplots(figsize=(16, 6))

bands = [
    (0.01, 0.4, "UV", "violet", "Ozone absorption\n(Aura/OMI)"),
    (0.4, 0.7, "Visible", "green", "True color imagery\n(MODIS, GOES, PACE)"),
    (0.7, 1.4, "Near-IR", "darkred", "Vegetation (NDVI)\n(MODIS, Landsat)"),
    (1.4, 3.0, "Short-wave IR", "maroon", "Cloud/ice\ndiscrimination"),
    (3.0, 15.0, "Thermal IR", "red", "Surface temp, OLR\n(ECOSTRESS, CERES)"),
    (15.0, 1000, "Far IR", "darkgray", "Atmospheric\nsounding (CrIS)"),
    (1e3, 1e6, "Microwave", "blue", "Precipitation radar\n(GPM, CloudSat)"),
    (1e6, 1e9, "Radio/Radar", "navy", "SAR, altimetry\n(JASON, GRACE)"),
]

for i, (wl_min, wl_max, name, color, sensors) in enumerate(bands):
    ax.barh(0, np.log10(wl_max) - np.log10(wl_min), left=np.log10(wl_min),
            height=0.6, color=color, alpha=0.6, edgecolor="black")
    mid = (np.log10(wl_min) + np.log10(wl_max)) / 2
    ax.text(mid, 0, name, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(mid, -0.55, sensors, ha="center", va="top", fontsize=7, style="italic")

ax.set_xlim(-2, 9.5)
ax.set_ylim(-1.2, 0.8)
ax.set_xlabel("log₁₀(Wavelength in μm)")
ax.set_title("Electromagnetic Spectrum - Remote Sensing Applications\nWhich satellites use which bands",
             fontsize=14)
ax.set_yticks([])

# Custom x-axis labels
tick_vals = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tick_labels = ["0.01μm", "0.1μm", "1μm", "10μm", "100μm", "1mm", "1cm", "10cm", "1m", "10m",
               "100m", "1km"]
ax.set_xticks(tick_vals)
ax.set_xticklabels(tick_labels, fontsize=8)
ax.grid(True, axis="x", alpha=0.3)
save_figure(fig, "16_em_spectrum_sensors.png")


# ================================================================
# 3. ORBITAL MECHANICS - Satellite orbits
# ================================================================
print("\n[3/7] Orbital Mechanics - Kepler's Laws & Orbit Types...")

def orbital_period(altitude_km):
    """Orbital period in minutes from altitude."""
    r = R_EARTH + altitude_km * 1000
    T = 2 * np.pi * np.sqrt(r**3 / (G * M_EARTH))
    return T / 60  # minutes

def orbital_velocity(altitude_km):
    """Orbital velocity in km/s."""
    r = R_EARTH + altitude_km * 1000
    return np.sqrt(G * M_EARTH / r) / 1000

altitudes = np.linspace(200, 42000, 500)
periods = [orbital_period(a) for a in altitudes]
velocities = [orbital_velocity(a) for a in altitudes]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Period vs altitude
axes[0].plot(altitudes, np.array(periods) / 60, "b-", linewidth=2)
axes[0].axhline(y=24, color="red", linestyle="--", alpha=0.7, label="24 hr (geostationary)")

# Mark actual satellites
satellites = {
    "ISS/ECOSTRESS": 408,
    "Aqua/MODIS": 705,
    "Suomi NPP": 824,
    "OCO-2": 705,
    "CALIPSO": 705,
    "CloudSat": 705,
    "GPM": 407,
    "PACE": 676,
    "Aura": 705,
    "GRACE-FO": 490,
    "JASON-3": 1336,
    "GOES-16": 35786,
}

for name, alt in satellites.items():
    period_hr = orbital_period(alt) / 60
    axes[0].plot(alt, period_hr, "ko", markersize=5)
    offset = (10, 5) if alt < 2000 else (-10, -15)
    axes[0].annotate(name, (alt, period_hr), textcoords="offset points",
                     xytext=offset, fontsize=7, arrowprops=dict(arrowstyle="-", alpha=0.3))

axes[0].set_title("Kepler's Third Law: T² ∝ a³\nOrbital Period vs Altitude", fontsize=13)
axes[0].set_xlabel("Altitude (km)")
axes[0].set_ylabel("Period (hours)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Orbit type diagram
ax2 = axes[1]
theta = np.linspace(0, 2 * np.pi, 100)

# Earth
earth = plt.Circle((0, 0), 1, color="lightblue", label="Earth")
ax2.add_patch(earth)

# Orbit rings (not to scale, schematic)
orbits = [
    (1.5, "LEO (200-2000km)", "green", "-"),
    (2.5, "MEO (2000-35786km)", "orange", "--"),
    (4.0, "GEO (35786km)", "red", "-"),
]
for radius, label, color, style in orbits:
    ax2.plot(radius * np.cos(theta), radius * np.sin(theta),
             color=color, linestyle=style, linewidth=2, label=label)

# HEO (elliptical)
a_heo, b_heo = 3.5, 1.5
ax2.plot(a_heo * np.cos(theta) - 1.5, b_heo * np.sin(theta),
         color="purple", linestyle=":", linewidth=2, label="HEO (elliptical)")

# Sun-synchronous annotation
ax2.annotate("Sun-synchronous\n(polar, LEO)\nMODIS, Suomi NPP,\nOCO-2, Aura, PACE",
             xy=(0.8, 1.2), fontsize=8,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

ax2.annotate("Geostationary\nGOES-16/17/18\n(fixed position)",
             xy=(2.5, 3.0), fontsize=8,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

ax2.set_xlim(-5.5, 5.5)
ax2.set_ylim(-5.5, 5.5)
ax2.set_aspect("equal")
ax2.set_title("Orbit Types (Schematic)", fontsize=13)
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(True, alpha=0.2)

fig.tight_layout()
save_figure(fig, "17_orbital_mechanics.png")

# Print key orbital facts
print("\n  === Satellite Orbits (test reference) ===")
for name, alt in sorted(satellites.items(), key=lambda x: x[1]):
    period = orbital_period(alt)
    vel = orbital_velocity(alt)
    orbit_type = ("LEO" if alt < 2000 else "MEO" if alt < 35786
                  else "GEO" if alt == 35786 else "HEO")
    print(f"  {name:15s}  alt={alt:6d}km  period={period:7.1f}min  "
          f"vel={vel:.1f}km/s  type={orbit_type}")


# ================================================================
# 4. IMAGING FIDELITY - GSD, IFOV, FOV
# ================================================================
print("\n[4/7] Imaging Fidelity - GSD, IFOV, Swath Width...")

fig, ax = plt.subplots(figsize=(14, 8))

# GSD = altitude * IFOV (in radians)
# Swath = altitude * FOV (in radians)

sensor_data = [
    # (name, altitude_km, GSD_m, swath_km)
    ("MODIS (Aqua)", 705, 250, 2330),
    ("VIIRS (Suomi NPP)", 824, 375, 3000),
    ("GOES-16 ABI", 35786, 500, "full disk"),
    ("ECOSTRESS", 408, 70, 53),
    ("GPM DPR", 407, 5000, 245),
    ("OCO-2", 705, 1290, 10.6),
    ("CERES", 824, 20000, "limb-to-limb"),
    ("PACE OCI", 676, 1000, 2663),
    ("CloudSat CPR", 705, 1400, 1.4),
    ("CALIPSO CALIOP", 705, 70, 0.070),
]

table_data = []
for name, alt, gsd, swath in sensor_data:
    if isinstance(gsd, (int, float)) and gsd > 0:
        ifov_urad = (gsd / (alt * 1000)) * 1e6  # microradians
    else:
        ifov_urad = "N/A"

    if isinstance(swath, (int, float)):
        fov_deg = np.degrees(swath * 1000 / (alt * 1000))
    else:
        fov_deg = swath

    table_data.append([name, f"{alt}", f"{gsd}", f"{swath}",
                       f"{ifov_urad:.0f}" if isinstance(ifov_urad, float) else ifov_urad,
                       f"{fov_deg:.1f}°" if isinstance(fov_deg, float) else fov_deg])

ax.axis("off")
table = ax.table(
    cellText=table_data,
    colLabels=["Instrument", "Altitude (km)", "GSD (m)", "Swath (km)",
               "IFOV (μrad)", "FOV"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(6)))
table.scale(1.2, 1.8)

# Color header
for j in range(6):
    table[0, j].set_facecolor("#4472C4")
    table[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Imaging Fidelity - Science Olympiad Satellites\n"
             "GSD = altitude × IFOV  |  Swath = altitude × FOV  |  "
             "Smaller GSD = finer detail", fontsize=13, pad=20)
save_figure(fig, "18_imaging_fidelity.png")


# ================================================================
# 5. PLANETARY EQUILIBRIUM TEMPERATURE
# ================================================================
print("\n[5/7] Planetary Energy Balance & Equilibrium Temperature...")

# Solar luminosity
L_sun = 4 * np.pi * R_SUN**2 * SIGMA * T_SUN**4
S0 = L_sun / (4 * np.pi * D_EARTH_SUN**2)  # Solar constant

print(f"\n  === Energy Balance Calculations ===")
print(f"  Solar luminosity:      {L_sun:.3e} W")
print(f"  Solar constant (S₀):   {S0:.1f} W/m²")

# Equilibrium temperature with varying albedo
albedos = np.linspace(0, 0.8, 100)
T_eq = ((S0 * (1 - albedos)) / (4 * SIGMA)) ** 0.25

# With greenhouse effect (simple model)
# T_surface ≈ T_eq * (1 + τ/2)^(1/4) for optical depth τ
T_greenhouse_1 = T_eq * (1 + 0.8) ** 0.25   # moderate greenhouse
T_greenhouse_2 = T_eq * (1 + 1.6) ** 0.25   # strong greenhouse

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Equilibrium temp vs albedo
axes[0].plot(albedos, T_eq - 273.15, "b-", linewidth=2, label="No atmosphere (bare rock)")
axes[0].plot(albedos, T_greenhouse_1 - 273.15, "orange", linewidth=2,
             label="Moderate greenhouse (τ=0.8)")
axes[0].plot(albedos, T_greenhouse_2 - 273.15, "r-", linewidth=2,
             label="Strong greenhouse (τ=1.6)")

# Mark actual Earth
earth_albedo = 0.30
T_eq_earth = ((S0 * (1 - earth_albedo)) / (4 * SIGMA)) ** 0.25
axes[0].plot(earth_albedo, T_eq_earth - 273.15, "ko", markersize=10)
axes[0].annotate(f"Earth (no atm)\nT={T_eq_earth-273.15:.1f}°C",
                 xy=(earth_albedo, T_eq_earth - 273.15),
                 xytext=(0.45, T_eq_earth - 273.15 + 10),
                 arrowprops=dict(arrowstyle="->"), fontsize=10)
axes[0].plot(earth_albedo, 15, "r*", markersize=15)
axes[0].annotate("Actual Earth\nT=15°C",
                 xy=(earth_albedo, 15),
                 xytext=(0.45, 25),
                 arrowprops=dict(arrowstyle="->"), fontsize=10)

axes[0].axhline(y=0, color="cyan", linestyle=":", alpha=0.5, label="Freezing point")
axes[0].set_title("Planetary Equilibrium Temperature vs Albedo\n"
                  "T = [S₀(1-α) / (4σ)]^(1/4)", fontsize=13)
axes[0].set_xlabel("Albedo (α)")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Energy balance diagram
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect("equal")

# Sun
sun = plt.Circle((1, 8), 0.8, color="yellow", ec="orange", linewidth=2)
ax2.add_patch(sun)
ax2.text(1, 8, "Sun", ha="center", fontsize=10)

# Earth
earth = plt.Circle((5, 3), 1.2, color="lightblue", ec="blue", linewidth=2)
ax2.add_patch(earth)
ax2.text(5, 3, "Earth", ha="center", fontsize=10)

# Atmosphere
atm = plt.Circle((5, 3), 2.0, color="lightyellow", ec="gray",
                  linewidth=1, linestyle="--", alpha=0.3)
ax2.add_patch(atm)
ax2.text(5, 5.3, "Atmosphere", ha="center", fontsize=9, style="italic")

# Arrows
# Incoming solar
ax2.annotate("", xy=(3.5, 4.5), xytext=(1.8, 7.2),
             arrowprops=dict(arrowstyle="->", color="orange", lw=2))
ax2.text(2.2, 6.2, f"S₀ = {S0:.0f}\nW/m²", fontsize=9, color="orange")

# Reflected
ax2.annotate("", xy=(1, 5.5), xytext=(3.5, 4.2),
             arrowprops=dict(arrowstyle="->", color="gold", lw=1.5))
ax2.text(1.5, 4.5, "Reflected\n(albedo α=0.30)\n~102 W/m²", fontsize=8, color="goldenrod")

# OLR (outgoing longwave)
ax2.annotate("", xy=(8, 7), xytext=(6.5, 4.5),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax2.text(7.5, 5.5, "OLR\n~239 W/m²", fontsize=9, color="red")

# Greenhouse back-radiation
ax2.annotate("", xy=(5.3, 2), xytext=(5.3, 4.5),
             arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
ax2.text(6, 2.8, "Greenhouse\nback-radiation\n~340 W/m²", fontsize=8, color="darkred")

# Energy imbalance
ax2.text(5, 0.5, "Current imbalance: ~1.0 W/m² (Earth gaining energy)\n"
         "This drives global warming", ha="center", fontsize=10,
         bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="red"))

ax2.axis("off")
ax2.set_title("Earth Energy Balance (Simplified)", fontsize=13)

fig.tight_layout()
save_figure(fig, "19_equilibrium_temperature.png")

print(f"  Earth equilibrium temp (no atmosphere): {T_eq_earth:.1f} K = {T_eq_earth-273.15:.1f}°C")
print(f"  Actual Earth surface temp:              288 K = 15°C")
print(f"  Greenhouse effect:                      +{288-T_eq_earth:.1f} K = +{15-(T_eq_earth-273.15):.1f}°C")


# ================================================================
# 6. RADIATIVE FORCING & FEEDBACK
# ================================================================
print("\n[6/7] Radiative Forcing & Climate Feedbacks...")

# CO2 forcing: ΔF = 5.35 × ln(C/C₀) W/m²
co2_levels = np.linspace(280, 800, 100)  # ppm
co2_preindustrial = 280  # ppm
forcing_co2 = 5.35 * np.log(co2_levels / co2_preindustrial)

# Temperature response: ΔT = λ × ΔF
# Climate sensitivity parameter λ ≈ 0.8 K/(W/m²) without feedbacks
# With feedbacks: λ ≈ 0.8 to 1.2 K/(W/m²)
lambda_no_fb = 0.3   # K/(W/m²) - Planck response only
lambda_with_fb = 0.8  # K/(W/m²) - including feedbacks

delta_T_no_fb = lambda_no_fb * forcing_co2
delta_T_with_fb = lambda_with_fb * forcing_co2

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# CO2 forcing
axes[0].plot(co2_levels, forcing_co2, "r-", linewidth=2)
axes[0].axvline(x=420, color="green", linestyle="--", alpha=0.7, label="Current (~420 ppm)")
axes[0].axvline(x=560, color="orange", linestyle="--", alpha=0.7, label="2× pre-industrial")
axes[0].set_title("CO₂ Radiative Forcing\nΔF = 5.35 × ln(C/C₀)", fontsize=13)
axes[0].set_xlabel("CO₂ Concentration (ppm)")
axes[0].set_ylabel("Radiative Forcing (W/m²)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Temperature response
axes[1].plot(co2_levels, delta_T_no_fb, "b-", linewidth=2,
             label="Planck response only (no feedbacks)")
axes[1].plot(co2_levels, delta_T_with_fb, "r-", linewidth=2,
             label="With feedbacks (water vapor, ice-albedo, etc.)")
axes[1].fill_between(co2_levels,
                     0.5 * forcing_co2,
                     1.2 * forcing_co2,
                     alpha=0.1, color="red", label="Uncertainty range")
axes[1].axvline(x=420, color="green", linestyle="--", alpha=0.7)
axes[1].axvline(x=560, color="orange", linestyle="--", alpha=0.7)
axes[1].set_title("Temperature Response to CO₂\nΔT = λ × ΔF", fontsize=13)
axes[1].set_xlabel("CO₂ Concentration (ppm)")
axes[1].set_ylabel("Temperature Change (°C)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("Radiative Forcing & Climate Sensitivity\n"
             "Key concept: logarithmic forcing means each doubling has same effect",
             fontsize=12, y=1.03)
fig.tight_layout()
save_figure(fig, "20_radiative_forcing.png")


# ================================================================
# 7. HEATING/COOLING RATES - Exponential & Piecewise Models
# ================================================================
print("\n[7/7] Heating/Cooling Rate Models...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

t = np.linspace(0, 200, 500)  # years

# Linear warming
rate_linear = 0.02  # °C/year
T_linear = 15 + rate_linear * t
axes[0].plot(t + 2025, T_linear, "r-", linewidth=2)
axes[0].set_title("Linear Model\nT(t) = T₀ + rt", fontsize=12)
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_ylim(14, 22)
axes[0].grid(True, alpha=0.3)
axes[0].text(2100, 16, f"Rate = {rate_linear}°C/yr\n"
             f"T(2100) = {15+rate_linear*75:.1f}°C", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Piecewise-linear (accelerating warming)
T_piecewise = np.piecewise(t, [t < 50, (t >= 50) & (t < 100), t >= 100],
                           [lambda x: 15 + 0.015 * x,
                            lambda x: 15.75 + 0.025 * (x - 50),
                            lambda x: 17.0 + 0.035 * (x - 100)])
axes[1].plot(t + 2025, T_piecewise, "r-", linewidth=2)
axes[1].axvline(x=2075, color="gray", linestyle=":", alpha=0.5)
axes[1].axvline(x=2125, color="gray", linestyle=":", alpha=0.5)
axes[1].set_title("Piecewise-Linear Model\nAccelerating warming rates", fontsize=12)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_ylim(14, 22)
axes[1].grid(True, alpha=0.3)
axes[1].text(2080, 16, "0.015°C/yr\n→ 0.025°C/yr\n→ 0.035°C/yr", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Exponential approach to new equilibrium
# T(t) = T_final - (T_final - T_0) * exp(-t/τ)
T_final = 18  # new equilibrium with doubled CO2
tau = 50  # thermal time constant (years)
T_exp = T_final - (T_final - 15) * np.exp(-t / tau)
axes[2].plot(t + 2025, T_exp, "r-", linewidth=2)
axes[2].axhline(y=T_final, color="red", linestyle="--", alpha=0.5,
                label=f"New equilibrium = {T_final}°C")
axes[2].axhline(y=15, color="blue", linestyle="--", alpha=0.5,
                label="Old equilibrium = 15°C")
axes[2].set_title("Exponential Approach to Equilibrium\n"
                  "T(t) = T_f - (T_f - T₀)e^(-t/τ)", fontsize=12)
axes[2].set_xlabel("Year")
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_ylim(14, 22)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)
axes[2].text(2100, 16, f"τ = {tau} years\n"
             f"63% of change\nin one τ", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

fig.suptitle("Section 3f: Heating/Cooling Rate Models\n"
             "Test may ask you to fit data, extrapolate, or calculate rates of change",
             fontsize=13, y=1.05)
fig.tight_layout()
save_figure(fig, "21_heating_cooling_models.png")


# ================================================================
# SUMMARY - Key Formulas Reference
# ================================================================
print("\n" + "=" * 60)
print("KEY FORMULAS FOR THE TEST")
print("=" * 60)
print("""
  BLACKBODY RADIATION:
    Wien's Law:           λ_peak = 2898 μm·K / T
    Stefan-Boltzmann:     P = σT⁴  (σ = 5.67×10⁻⁸ W/m²/K⁴)
    Rayleigh-Jeans:       B(λ,T) ≈ 2ckT/λ⁴  (long wavelength approx)

  ORBITAL MECHANICS:
    Kepler's 3rd Law:     T² = (4π²/GM)a³
    Orbital velocity:     v = √(GM/r)
    Geostationary alt:    35,786 km  (T = 24 hr)

  IMAGING:
    GSD = altitude × IFOV
    Swath width = altitude × FOV
    Angular resolution = λ / D (diffraction limit)

  ENERGY BALANCE:
    Solar constant:       S₀ ≈ 1361 W/m²
    Absorbed solar:       (S₀/4)(1-α)  (α = albedo ≈ 0.30)
    Equilibrium temp:     T = [S₀(1-α)/(4σ)]^(1/4) ≈ 255 K
    Greenhouse effect:    ΔT ≈ +33°C  (255K → 288K)

  RADIATIVE FORCING:
    CO₂ forcing:          ΔF = 5.35 × ln(C/C₀)  W/m²
    Temperature response: ΔT = λ × ΔF
    Climate sensitivity:  ~3°C per doubling of CO₂

  CLIMATE:
    ENSO: El Niño = warm Nino 3.4 SST anomaly > +0.5°C
          La Niña = cool Nino 3.4 SST anomaly < -0.5°C
""")

print("=" * 60)
print(f"All study diagrams saved to ./output/")
print("=" * 60)
