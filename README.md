# Phase-Portrait-Visualizer
A Phase Portrait graphing tool that includes user input and uses either a matrix representation or a system of equations of a dynamical system.

The left graph is for 2 dimensional systems and the right graph is for 3 dimensional systems. The number of orbits can be decided by the user but has a default
value of 5 and is not required. Though it should be kept in mind that trajectories may not always appear as inputed value, as arithmetic may cause null values or 
uninterpretable orbits.

---Input should be entered in the following format---
All input should be entered into appropriate boxes before submit is clicked. Once submit is clicked by user, whatever is currently in the boxes will be graphed.

For matrix representation (upper left input boxes), one matrix at a time should be entered in the following format:
2x2: for a matrix [[a, c], [b, d]], |a, b| , this matrix should be entered as [a, b, c, d], or a, b, c, d. Commas are required
                                    |c, d| 
     between elements
3x3: for a matrix [[a, d, g], [b, e, h], [c, f, i]], |a, b, c| , this matrix should be entered as [a, b, c, d, e, f, g, h, i], or a, b, c, d, e, f, g, h, i. Commas are required
                                                     |d, e, f| 
                                                     |g, h, i|
     between elements
For a system of equations (upper right input boxes) the equations should be entered in the following format:
all variables should be lower case x, y, and z only. These equations should use standard mathematical operators and for powers, the format should be base^exponent

The bottom left box is for the number of orbits, and should hold the desired number of orbits as given by the user before the user clicks submit. This value must be an integer
or an error will occur
