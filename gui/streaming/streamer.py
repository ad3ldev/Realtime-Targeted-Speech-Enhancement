import time
import math
import torch

def _valid_length(length, depth, resample, kernel_size, stride):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * resample)
        for idx in range(depth):
            length = math.ceil((length - kernel_size) / stride) + 1
            length = max(length, 1)
        for idx in range(depth):
            length = (length - 1) * stride + kernel_size
        length = int(math.ceil(length / resample))
        return int(length)

class Streamer:
    """
    Streaming implementation for Demucs. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.

    Args:
        - demucs (Demucs): Demucs model.
        - dry (float): amount of dry (e.g. input) signal to keep. 0 is maximum
            noise removal, 1 just returns the input signal. Small values > 0
            allows to limit distortions.
        - num_frames (int): number of frames to process at once. Higher values
            will increase overall latency but improve the real time factor.
        - resample_lookahead (int): extra lookahead used for the resampling.
        - resample_buffer (int): size of the buffer of previous inputs/outputs
            kept for resampling.
    """
    def __init__(self,
                 model,
                 embedding,
                 dry=0.0,
                 num_frames=1,
                 resample_lookahead=64,
                 resample_buffer=256,
                 chin=1,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 resample=1,
                 normalize=True,
                 floor=1e-3,
                 device = 'cpu'):
        # self.demucs = demucs
        self.model = model
        self.embedding = embedding
        self.chin = chin
        self.normalize = normalize
        self.floor = floor
        self.device = device
        self.enhance = True
        
        total_stride = stride ** depth // resample
        self.lstm_state = None
        self.conv_state = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead
        resample_buffer = min(total_stride, resample_buffer)
        self.resample_buffer = resample_buffer
        self.frame_length = _valid_length(1, depth=depth, resample=resample, kernel_size=kernel_size, stride=stride) + total_stride * (num_frames - 1)
        self.total_length = self.frame_length + self.resample_lookahead
        self.stride = total_stride * num_frames
        self.resample = resample
        self.resample_in = torch.zeros(chin, resample_buffer, device=device)
        self.resample_out = torch.zeros(chin, resample_buffer, device=device)

        self.frames = 0
        self.total_time = 0
        self.variance = 0
        self.pending = torch.zeros(chin, 0, device=device)

        # bias = demucs.decoder[0][2].bias
        # weight = demucs.decoder[0][2].weight
        # chin, chout, kernel = weight.shape
        # self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)
        # self._weight = weight.permute(1, 2, 0).contiguous()

    def reset_time_per_frame(self):
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self):
        return self.total_time / self.frames

    def flush(self):
        """
        Flush remaining audio by padding it with zero and initialize the previous
        status. Call this when you have no more input and want to get back the last
        chunk of audio.
        """
        self.lstm_state = None
        self.conv_state = None
        pending_length = self.pending.shape[1]
        padding = torch.zeros(self.chin, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav):
        """
        Apply the model to mix using true real time evaluation.
        Normalization is done online as is the resampling.
        """
        begin = time.time()
        resample_buffer = self.resample_buffer
        stride = self.stride
        resample = self.resample

        if wav.dim() != 2:
            raise ValueError("input wav should be two dimensional.")
        chin, _ = wav.shape
        if chin != self.chin:
            raise ValueError(f"Expected {self.chin} channels, got {chin}")

        self.pending = torch.cat([self.pending, wav], dim=1)
        outs = []
        while self.pending.shape[1] >= self.total_length:
            self.frames += 1
            frame = self.pending[:, :self.total_length]
            dry_signal = frame[:, :stride]
            if self.normalize:
                mono = frame.mean(0)
                variance = (mono**2).mean()
                self.variance = variance / self.frames + (1 - 1 / self.frames) * self.variance
                frame = frame / (self.floor + math.sqrt(self.variance))
            padded_frame = torch.cat([self.resample_in, frame], dim=-1)
            self.resample_in[:] = frame[:, stride - resample_buffer:stride]
            frame = padded_frame

            # if resample == 4:
            #     frame = upsample2(upsample2(frame))
            # elif resample == 2:
            #     frame = upsample2(frame)
            frame = frame[:, resample * resample_buffer:]  # remove pre sampling buffer
            frame = frame[:, :resample * self.frame_length]  # remove extra samples after window

            out = self._separate_frame(frame)
            padded_out = torch.cat([self.resample_out, out], 1)
            self.resample_out[:] = out[:, -resample_buffer:]
            # if resample == 4:
            #     out = downsample2(downsample2(padded_out))
            # elif resample == 2:
            #     out = downsample2(padded_out)
            # else:
            out = padded_out

            out = out[:, resample_buffer // resample:]
            out = out[:, :stride]

            if self.normalize:
                out *= math.sqrt(self.variance)
            out = self.dry * dry_signal + (1 - self.dry) * out
            outs.append(out)
            self.pending = self.pending[:, stride:]

        self.total_time += time.time() - begin
        if outs:
            out = torch.cat(outs, 1)
        else:
            out = torch.zeros(chin, 0, device=wav.device)
        return out

    def _separate_frame(self, frame):
        raise NotImplementedError

class PDenoiserStreamer(Streamer):
    def __init__(self, model, embedding, dry=0.0, num_frames=1, resample_lookahead=64, resample_buffer=256, chin=1, depth=5, kernel_size=8, stride=4, resample=1, normalize=True, floor=1e-3, device = 'cpu'):
        super(PDenoiserStreamer, self).__init__(model, embedding, dry, num_frames, resample_lookahead, resample_buffer, chin, depth, kernel_size, stride, resample, normalize, floor, device)

    def _separate_frame(self, frame):
        return self.model(frame, self.embedding)[0]

class PFFSNStreamer(Streamer):
    def __init__(self, model, embedding, dry=0.0, num_frames=1, resample_lookahead=64, resample_buffer=256, chin=1, depth=5, kernel_size=8, stride=4, resample=1, normalize=True, floor=1e-3, device = 'cpu'):
        super(PFFSNStreamer, self).__init__(model, embedding, dry, num_frames, resample_lookahead, resample_buffer, chin, depth, kernel_size, stride, resample, normalize, floor, device)

    def _separate_frame(self, frame):
        return self.model({"noisy": frame, "reference_subbands": self.embedding})[0]
