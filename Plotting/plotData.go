package plotting

import (
	"image/color"
	"log"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

func PlotData(data [][]float64, color []float64, cmap map[float64]color.Color, path string) {
	p := plot.New()

	for key, element := range cmap {
		xys := plotter.XYs{}

		for i := range data {
			if color[i] == key {
				xys = append(xys, plotter.XY{X: data[i][0], Y: data[i][1]})
			}
		}

		scatter, err := plotter.NewScatter(xys)
		handleErr(err)
		scatter.GlyphStyle.Color = element
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}

		p.Add(scatter)
	}

	wt, err := p.WriterTo(500, 400, "png")
	handleErr(err)

	f, err := os.Create(path)
	handleErr(err)

	defer func() {
		err = f.Close()
		handleErr(err)
	}()

	wt.WriteTo(f)
}

func handleErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
