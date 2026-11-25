# Matplotlib experiment labs

This folder contains small, independent Matplotlib experiments for learning
how to create and update plots in different scenarios. The examples are
fragmented by purpose rather than organized as one large application, so each
script can be studied or reused as a self-contained plotting reference.

## Examples

### dynamic_scatter.py

This example demonstrates how to draw and refresh a scatter plot dynamically.
It uses NumPy to generate random point coordinates and Matplotlib to update a
fixed two-dimensional coordinate system repeatedly.

Key techniques:

- Create a fixed plot area with `plt.figure()`
- Set the visible `x` and `y` ranges with `plt.xlim()` and `plt.ylim()`
- Clear the previous frame with `plt.cla()`
- Draw a scatter point with `plt.scatter()`
- Display and pause between updates with `plt.draw()` and `plt.pause()`
- Keep the final window open with `plt.show(block=True)`