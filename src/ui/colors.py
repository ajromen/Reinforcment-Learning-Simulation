import darkdetect
import time
import threading

IS_LIGHT_MODE = darkdetect.isLight()

if darkdetect.isDark():
    background_primary = "#222222"
    background_secondary = "#191919"
    foreground = "#ffffff"
    foreground_secondary = "#9C9C9C"
    background_dots = "#353535"
    muscle = "#612F2F"
    bone = "#979797"
    light_background = "#D9D9D9"

    # RGB

    bone_rgb = [151, 151, 151]
    bone_rbg_semi_transparent = [151, 151, 151, 60]
    foreground_rgb = [255, 255, 255]
    muscle_rgb_semi_transparent = [97, 47, 47, 80]

else:
    background_primary = "#DDDDDD"
    background_secondary = "#E6E6E6"
    foreground = "#000000"
    foreground_secondary = "#404040"
    background_dots = "#CACACA"
    muscle = "#903B3B"
    bone = "#A1A1A1"
    light_background = "#404040"

    # RGB
    bone_rgb = [161, 161, 161]
    bone_rbg_semi_transparent = [161, 161, 161, 60]
    foreground_rgb = [0, 0, 0]
    muscle_rgb_semi_transparent = [144, 59, 59, 80]


