# CaloClouds Model names decoded

There are 4 varients of CaloClouds present.

1. The one that was automatically downloaded with the package. This is just called `cc`.
2. One that was built using the same architecture as the first. Only the flow model ever changes, and the original is 10 original layers, with all 65 values predicted. This is `ccO`
3. One that was built with the lean architecture. So the flow model has only 4 layers, they are alt1 layers, and it predicts only 63 values. This is `ccA4`.
4. One that was build with a very lean architecture. So the flow model has only 2 layers, they are alt1 layers and it predicts only 63 values. This is `ccA2`.
