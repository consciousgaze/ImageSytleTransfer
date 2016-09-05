#/usr/bin/env python


def getStyleGradientFunction(origLayers):
    def getStyleGradient(generated):
        ...
    return getStyleGradient

def getContentGradientFunction(tgtLayers):
    def getContentGradient(generated):
        ...
    return getContentGradient

def getStyleLossFunction(origLayers):
    def getStyleLoss(generated):
        pass
    return getStyleLoss

def getContentLossFunction(tgtLayers):
    def getContentLoss(genertated):
        ...
    return getContentLoss

def styleConvert(generated, getContentLoss, getStyleLoss,
                 getStyleGradient, getContentGradient,
                 alpha = 0.1, threshold = 0.01):
    previousLoss = None
    currentLoss = None
    for loop in range(3000):
        if not (previousLoss == None or (previousLoss - currentLoss)/previousLoss < threshold):
            break

        styleGradient = getSytleGradient(generated)
        contentGradient = getContentGradient(generated)

        generated -= (alpha*styleGradient + contentGradient) * generated

        previousLoss = currentLoss
        currentLoss = alpha*getStyleLoss(generated) + getContentLoss(generated)

    return generated


if __name__ == "__main__":
    orgLayers = getConvLayers(originImage)
    tgtLayers = getConvLayers(targetImage)

    generated = generateInitImage()

    getContentLoss = getContentLossFunction(tgtLayers)
    getContentGradient = getContentGradientFunction(tgtLayers)

    getStyleLoss = getStyleLossFunction(orgLayers)
    getStyleGradient = getStyleGradientFunction(orgLayers)

    generated = styleConvert(generated, getContentLoss, getStyleLoss,
                             getStyleGradient, getContentGradient)

    show(genertated)

