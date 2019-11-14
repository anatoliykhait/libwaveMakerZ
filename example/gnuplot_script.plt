set terminal png size 800,600
set termoption dash

set output 'elevation.png'

set xrange [0:3]

set ylabel "Elevation [m]"
set xlabel "time [s]"
plot "time_series" using 1:2 with lines linewidth 4 title "free", \
     "time_series" using 1:3 with lines linewidth 4 title "bound 2nd order"

set output 'wavemaker_displacement.png'

set xrange [0:3]

set ylabel "Wavemaker Displacement [m]"
set xlabel "time [s]"
plot "time_series" using 1:4 with lines linewidth 4 title "1st order", \
     "time_series" using 1:5 with lines linewidth 4 title "2nd order - displacement correction", \
     "time_series" using 1:6 with lines linewidth 4 title "2nd order - bound waves correction"

set output 'spectrum_elevation.png'

set xrange [0:6]

set ylabel "Elevation Amplitude [m]"
set xlabel "Angular Frequency [rad/s]"
plot "spectra" using 1:4 with lines linewidth 4 title "free", \
     "spectra" using 1:5 with lines linewidth 4 title "bound 2nd order"

set output 'spectrum_wavemaker_displacement.png'

set xrange [0:6]

set ylabel "Wavemaker Displacement Amplitude [m]"
set xlabel "Angular Frequency [rad/s]"
plot "spectra" using 1:6 with lines linewidth 4 title "1st order", \
     "spectra" using 1:7 with lines linewidth 4 title "2nd order - displacement correction", \
     "spectra" using 1:8 with lines linewidth 4 title "2nd order - bound waves correction"