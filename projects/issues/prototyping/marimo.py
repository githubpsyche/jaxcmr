import marimo

__generated_with = "0.7.0"
app = marimo.App()


@app.cell
def __():
    import debugpy; debugpy.listen(5678); debugpy.wait_for_client()
    return debugpy,


@app.cell
def computes_z(x, y):
    z = x + y + 1
    z
    return z,


@app.cell
def defines_y(x):
    y = x + 1
    breakpoint()
    y
    return y,


@app.cell
def defines_x():
    x = 2
    return x,


if __name__ == "__main__":
    app.run()
