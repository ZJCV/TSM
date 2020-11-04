
# Model Zoo

## [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-jqnz{background-color:#F3F6F6;color:#404040;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-rt64{background-color:#F3F6F6;color:#9B59B6;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj"><span style="font-weight:bold">config</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">resolution(TxHxW)</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">gpus</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">backbone</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">pretrain</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">top1 acc</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">top5 acc</span></th>
    <th class="tg-uzvj"><span style="font-weight:bold">testing protocol</span></th>
    <th class="tg-wa1i"><span style="font-weight:bold">inference_time(video/s)</span></th>
    <th class="tg-wa1i"><span style="font-weight:bold">gpu_mem(M)</span></th>
    <th class="tg-wa1i"><span style="font-weight:bold">ckpt</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-rt64"><a href="https://cloud.zhujian.tech:9300/s/MwAMXHsXQgAZRwD" target="_blank" rel="noopener noreferrer">tsn_r50_ucf101_rgb_raw_dense_1x16x4</a></td>
    <td class="tg-jqnz">4x256x256</td>
    <td class="tg-jqnz">2</td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">tsn</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">ImageNet</span></td>
    <td class="tg-jqnz">80.881</td>
    <td class="tg-jqnz"><span style="font-weight:400;font-style:normal">95.48</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">1 clips x 1 crop</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">x</span></td>
    <td class="tg-jqnz"><span style="background-color:#F3F6F6">x</span></td>
    <td class="tg-rt64"><a href="https://cloud.zhujian.tech:9300/s/ZKXim94beK4a9EJ">ckpt</a></td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td class="tg-0wh7"><a href="https://cloud.zhujian.tech:9300/s/bY7jRPpAD9mKqkW" target="_blank" rel="noopener noreferrer">tsn_r50_ucf101_rgb_raw_seg_1x1x3</a></td>
    <td class="tg-tiqg">4x256x256</td>
    <td class="tg-tiqg">2</td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">tsn</span></td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">imagenet</span></td>
    <td class="tg-tiqg">81.589</td>
    <td class="tg-tiqg"><span style="font-weight:400;font-style:normal">95.964</span></td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">1 clips x 1 crop</span></td>
    <td class="tg-tiqg"><span style="background-color:#F3F6F6">x</span></td>
    <td class="tg-tiqg">x</td>
    <td class="tg-0wh7"><a href="https://cloud.zhujian.tech:9300/s/xqbSpLFcQkJADbz" target="_blank" rel="noopener noreferrer">ckpt</a></td>
  </tr>
</tbody>
</table>


[10/30 06:28:21][INFO] trainer.py:  85: iter: 045960, lr: 0.00040, loss: 0.022337 (0.362282), tok1: 99.167 (90.161), tok5: 100.000 (97.122), time: 0.425 (0.484), eta: 0:32:34, mem: 7884M
[10/30 06:28:26][INFO] trainer.py:  85: iter: 045970, lr: 0.00040, loss: 0.026959 (0.362210), tok1: 100.000 (90.164), tok5: 100.000 (97.123), time: 0.425 (0.484), eta: 0:32:29, mem: 7884M
[10/30 06:28:30][INFO] trainer.py:  85: iter: 045980, lr: 0.00040, loss: 0.057488 (0.362143), tok1: 98.333 (90.165), tok5: 100.000 (97.124), time: 0.424 (0.484), eta: 0:32:24, mem: 7884M
[10/30 06:28:34][INFO] trainer.py:  85: iter: 045990, lr: 0.00040, loss: 0.028552 (0.362071), tok1: 99.167 (90.167), tok5: 100.000 (97.124), time: 0.426 (0.484), eta: 0:32:19, mem: 7884M


.514), tok5: 100.000 (98.620), time: 0.421 (0.432), eta: 0:08:25, mem: 8387M
[10/30 23:46:11][INFO] trainer.py:  85: iter: 018840, lr: 0.00092, loss: 0.005178 (0.141351), tok1: 99.896 (96.516), tok5: 100.000 (98.620), time: 0.420 (0.432), eta: 0:08:21, mem: 8387M
[10/30 23:46:15][INFO] trainer.py:  85: iter: 018850, lr: 0.00092, loss: 0.012340 (0.141282), tok1: 99.896 (96.518), tok5: 100.000 (98.621), time: 0.391 (0.432), eta: 0:08:16, mem: 8387M
[10/30 23:46:19][INFO] trainer.py:  85: iter: 018860, lr: 0.00092, loss: 0.005033 (0.141210), tok1: 100.000 (96.519), tok5: 100.000 (98.622), time: 0.421 (0.432), eta: 0:08:12, mem: 8387M
[10/30 23:46:23][INFO] trainer.py:  85: iter: 018870, lr: 0.00092, loss: 0.004718 (0.141138), tok1: 100.000 (96.521), tok5: 100.000 (98.623), time: 0.423 (0.432), eta: 0:08:08, mem: 8387M
[10/30 23:46:27][INFO] trainer.py:  85: iter: 018880, lr: 0.00092, loss: 0.004489 (0.141065), tok1: 100.000 (96.523), tok5: 100.000 (98.623), time: 0.421 (0.432), eta: 0:08:03