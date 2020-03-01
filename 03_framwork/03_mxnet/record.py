import mxnet as mx

# 创建并读取随机索引的rec文件
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
for i in range(5):
    record.write_idx(i, b'record_%d' % i)
record.close()

record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
record.read_idx(3)
var = record.keys

print(var)
