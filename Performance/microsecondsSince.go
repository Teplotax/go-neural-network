package Performance

import (
	"fmt"
	"time"
)

func MicrosecondsSince(start time.Time) {
	duration := time.Since(start)
	fmt.Println("Execution time: ", duration.Microseconds(), "μs")
}
