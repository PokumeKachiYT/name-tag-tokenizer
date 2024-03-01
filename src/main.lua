require('nn')
require('torch')

local net = nn.Sequential()

net:add(nn.SpatialConvolution(1,6,320,270))
net:add
