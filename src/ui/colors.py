import darkdetect
import time
import threading

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


