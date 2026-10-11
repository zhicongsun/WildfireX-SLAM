"""Scene-control constants (no simulator dependency, safe to import from config validation)."""

# Sun (DirectionalLight) pitch presets in degrees; negative = above the horizon.
# Measured in the Electric Dreams level: below ~-15 deg the ground goes black. In WildfireX-Sim 1.0 exposure is
# locked by the level's post-process volumes; 1.1 adds a moon and lowers the auto-exposure bound for night.
SUN_PRESETS = {"noon": -75.0, "morning": -30.0, "dusk": -22.0, "night": 20.0}
SUN_YAW_DEG = 130.0
# dusk: the sun sets on the opposite side from where it rose (with the morning azimuth, low-sun sites lay in terrain shadow)
SUN_YAW_PRESETS = {"dusk": 310.0}
FIRE_HIDE_DROP = 2000.0  # m; "fire off" moves the Niagara actors underground (scaling does not hide particles)

# WildfireX-Sim 1.1 moon (second atmosphere light)
MOON_PITCH_DEG = -40.0       # above the horizon when the "night" preset is used
MOON_PARK_PITCH_DEG = 30.0   # below the horizon otherwise
MOON_YAW_DEG = -50.0

FLAME_VARIANTS = ("NFire_Grd_00", "NFire_Grd_01", "NFire_Grd_02", "NFire_00", "NFire_03", "NFire_06", "NFire_09", "NFire_12")
FIRE_KITS = ("A", "B", "C")

# Fire-kit point light ("glow"), placed automatically only for these presets (too strong in daylight).
# simSetLightIntensity does not work on the kit lights, so the glow is dimmed by burying the light (it casts no
# shadows): value = metres below the ground; ground irradiance falls with the squared distance.
# Measured at ed04: light 3 m above ground made dusk and night look the same (auto exposure turns everything orange);
# 30 m below gives a dark night lit around the fire, 45 m a faint warm glow at dusk.
GLOW_DEPTH_M = {"dusk": 45.0, "night": 30.0}
GLOW_FULL_DEPTH_M = -3.0  # fire.glow: true -> light 3 m above the ground (full strength)
