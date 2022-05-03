nscore_par = '''\
                  Parameters for NSCORE
                  *********************

START OF PARAMETERS:
{sondcomp_data}      -file with data
{data_col}   0           -  columns for variable and weight
-1.0e21   1.0e21         -  trimming limits
0                        -1=transform according to specified ref. dist.
../histsmth/histsmth.out -  file with reference dist.
1   2                    -  columns for variable and weight
{nscore_out}               -file for output
{nscore_trn}               -file for output transformation table'''

sgsim_par = '''\
                  Parameters for SGSIM
                  ********************

START OF PARAMETERS:
{nscore_out}           -file with data
{x}  {y}  {z}  {data_col}  0  0  -  columns for X,Y,Z,vr,wt,sec.var.
-1.0e21       1.0e21             -  trimming limits
1                             -transform the data (0=no, 1=yes)
{nscore_trn}                  -  file for output trans table
0                             -  consider ref. dist (0=no, 1=yes)
histsmth.out                  -  file with ref. dist distribution
0  0                          -  columns for vr and wt
{Zmin}    {Zmax}              -  zmin,zmax(tail extrapolation)
1       0.0                   -  lower tail option, parameter
1      15.0                   -  upper tail option, parameter
0                             -debugging level: 0,1,2,3
sgsim.dbg                     -file for debugging output
{sgsim_out}                   -file for simulation output
{realizations}                -number of realizations to generate
{nx}    {ox}    {xsize}       -nx,xmn,xsiz
{ny}    {oy}    {ysize}       -ny,ymn,ysiz
{nz}    {oz}    {zsize}       -nz,zmn,zsiz
{hull_file}                   -file with convex hull
{random}                      -random number seed
{mindat}     {maxdat}         -min and max original data for sim
{nodes}                       -number of simulated nodes to use
1                             -assign data to nodes (0=no, 1=yes)
0     3                       -multiple grid search (0=no, 1=yes),num
{octant_data}                 -maximum data per octant (0=not used)
{hmax}  {hmin}  {vert}        -maximum search radii (hmax,hmin,vert)
 {azim}   {dip}   {plunge}    -angles for search ellipsoid
23    31    33                -size of covariance lookup table
0     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC
../data/ydata.dat             -  file with LVM, EXDR, or COLC variable
0                             -  column for secondary variable
{nst_sim}    {nugget_sim}             -nst, nugget effect
{model_par_sim}'''


kt3d_par = '''\
                  Parameters for KT3D
                  *******************

START OF PARAMETERS:
{sondcomp_data}              -file with data
0  {x}  {y}  {z}  {data_col}  0                 -   columns for DH,X,Y,Z,var,sec var
-1.0e21   1.0e21                 -   trimming limits
0                                -option: 0=grid, 1=cross, 2=jackknife
xvk.dat                          -file with jackknife data
1   2   0    3    0              -   columns for X,Y,Z,vr and sec var
0                                -debugging level: 0,1,2,3
kt3d.dbg                         -file for debugging output
{kt3d_out}                         -file for kriged output
{nx}    {ox}    {xsize}                  -nx,xmn,xsiz
{ny}    {oy}    {ysize}                  -ny,ymn,ysiz
{nz}    {oz}    {zsize}                  -nz,zmn,zsiz
2    2      1                    -x,y and z block discretization
{mindat}     {maxdat}                           -min, max data for kriging
{octant_data}                                -max per octant (0-> not used)
{hmax}  {hmin}  {vert}                 -maximum search radii
 {azim}   {dip}   {plunge}                 -angles for search ellipsoid
1     2.302                      -0=SK,1=OK,2=non-st SK,3=exdrift
0 0 0 0 0 0 0 0 0                -drift: x,y,z,xx,yy,zz,xy,xz,zy
0                                -0, variable; 1, estimate trend
extdrift.dat                     -gridded file with drift/mean
4                                -  column number in gridded file
{nst_krig}    {nugget_krig}                   -nst, nugget effect
{model_par_krig}'''