

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=6):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        slots = torch.randn(C, 3)
        slots = self._l2norm(slots, dim=1)
        self.register_buffer('slots', slots)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.permute([0,2,3,1])

        x = f.reshape([-1,c])

        slots = slots.to(device)
        with torch.no_grad():
            for i in range(10):
                z = torch.matmul(f, slots)      # NxK
                z = F.softmax(z, dim=-1)                 # NxK
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                slots = torch.matmul(f.T, z_)       # CxK
                slots = _l2norm(slots, dim=0)


        # attn_logits = torch.matmul(f, slots)
        # attn = attn_logits.softmax(dim=-1)
        # print(attn.shape)
        # attn = attn.reshape([B,H,W,3])
        # attn = attn.permute([0,3,1,2])
                
        # !!! The moving averaging operation is writtern in train.py, which is significant.


        z_t = z.permute(1, 0)            # k * n
        x = mu.matmul(z_t)                  # c * n
        x = x.permute([1,0])
        x = x.reshape([b, h, w, c ])
        x = x.permute([0,3,1,2])              # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, slots