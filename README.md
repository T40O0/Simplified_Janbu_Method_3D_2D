# Slope Stability: Simplified Janbu Method  

 - This is a Python port of the following MATLAB code.  
([Landslide-Forensics](https://github.com/benalesh/Landslide-Forensics))
Bunn, M., Leshchinsky, B., & Olsen, M. J. (2020). Geologic trends in shear strength properties inferred through three-dimensional back analysis of landslide inventories. Journal of Geophysical Research: Earth Surface, 125, https://doi.org/10.1029/2019JF005461
 - 2D back-and-forward analysis added.

## Requirements
This project requires the following Python packages (with versions tested):
 - geopandas  1.0.1
 - matplotlib 3.10.1
 - numpy      2.2.4
 - pandas     2.2.3
 - rasterio   1.4.3
 - scipy      1.15.2
 - shapely    2.1.0

## Features
 - The slip direction and c or φ can be obtained by back-calculation using the simplified 3D Janbu method.
 - 2D inverse calculation is possible for the slip direction calculated in 3D.
 - Forward calculation in 2D is possible for the slip direction, c and φ calculated in 3D.
 - Saves c, φ, safety factor FS and slip direction as histogram, csv and shapefile.

## Acknowledgments
Special thanks to Michael Bunn, Ben Leshchinsky, and Michael J. Olsen, whose work (Bunn et al., 2020) laid the foundation for this code. Their original research and MATLAB code were  invaluable in the development of this Python port.

## Licence
This project is licensed under the MIT licence - see the [LICENSE file](LICENSE) for details.

