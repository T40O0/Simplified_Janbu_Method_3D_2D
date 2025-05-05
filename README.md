# Slope Stability: Simplified Janbu Method (3D & 2D)  

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
 - The slip direction and c or φ can be calculated by 3D back analysis using the simplified Janbu method.
 - 2D back calculation is possible for the slip direction calculated in 3D.
 - Forward calculation in 2D is possible for the slip direction, c and φ calculated in 3D.
 - Saves c, φ, FS and slip direction as histogram, csv and shapefile.

## References
Ugai, K., and Hosobori, K. (1988). Extension of simplified Bishop method, simplified Janbu method and Spencer method to three-dimensions. Proc., Japanese Soc. of Civ. Engrs., Tokyo, 394/III-9, 21–26 (in Japanese).  
Hungr, O., Salgado, F. M., & Byrne, P. M. (1989). Evaluation of a three-dimensional method of slope stability analysis. Canadian Geotechnical Journal, 26(4), 679–686.  
O. Hungr (1987). An extension of Bishop's simplified method of slope stability analysis to three dimensions. Géotechnique, 37(1), 113-117.

## Acknowledgments
Special thanks to Michael Bunn, Ben Leshchinsky, and Michael J. Olsen, whose work (Bunn et al., 2020) laid the foundation for this code. Their original research and MATLAB code were  invaluable in the development of this Python port.

## Licence
This project is licensed under the MIT licence - see the [LICENSE file](LICENSE) for details.

