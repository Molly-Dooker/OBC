import math
import time
from tqdm import tqdm
import torch
import torch.nn as nn

from quant import *

import ipdb
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEBUG = False 


class TrueOBS:

    def __init__(self, layer, rel_damp=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # Accumulate in double precision
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.double)
        self.nsamples = 0
        self.rel_damp = rel_damp

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def invert(self, H):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            print('Hessian not full rank.')
            tmp = 1 * torch.eye(self.columns, device=self.dev)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare(self, columnslast=False):
        if columnslast: 
            perm = torch.arange(self.columns, device=self.dev)
            if len(self.layer.weight.shape) == 4:
                perm = perm.reshape(list(self.layer.weight.shape)[1:])
                perm = perm.permute([1, 2, 0])
                perm = perm.flatten()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        H = self.H.float()
        if self.rel_damp > 0:
            damp = self.rel_damp * torch.diag(H).mean()
            H += damp * torch.eye(H.shape[0], device=self.dev)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if columnslast:
            H = H[perm, :][:, perm]
            W = W[:, perm]
        Hinv = self.invert(H)
        Losses = torch.zeros([self.rows, self.columns + 1], device=self.dev)
        if columnslast:
            return W, H, Hinv, Losses, perm
        return W, H, Hinv, Losses

    def prepare_iter(self, i1, parallel, W, Hinv1):
        i2 = min(i1 + parallel, self.rows)
        count = i2 - i1
        w = W[i1:i2, :]
        Hinv = Hinv1.unsqueeze(0).repeat((count, 1, 1))
        mask = torch.zeros_like(w).bool()
        rangecount = torch.arange(count, device=self.dev)
        idxcount = rangecount + i1
        return i2, count, w, Hinv, mask, rangecount, idxcount

    def prepare_sparse(self, weight, mask, Hinv, H):
        start = int(torch.min(torch.sum((weight == 0).float(), 1)).item()) + 1
        # 행별 weight 값 0 인 요소 개수들의 min 값+1
        for i in range(weight.shape[0]):
            tmp = weight[i] == 0
            H1 = H.clone()
            H1[tmp, :] = 0
            H1[:, tmp] = 0
            H1[tmp, tmp] = 1
            Hinv[i] = self.invert(H1)
            mask[i, torch.nonzero(tmp, as_tuple=True)[0][:(start - 1)]] = True
            # hessian 은 w==0 인걸 반영했지만 mask 는 그대로 반영안하고 start 에 기반해서 반영함
        return start

    def quantize(self, parallel=32):
        W, H, Hinv1, Losses = self.prepare() # 레이어별로 H, Hinv 구함 W 는  원 shape 사용 1000,512     
        Q = torch.zeros_like(W)
        self.quantizer.find_params(W, weight=True) # 일단 간단한 ptq 로 weight에 대한  scale 과 zp 구함 sym이므로 zp=0, absmax->mse 
        parallel = 32
        for i1 in tqdm(range(0, self.rows, parallel),desc='batch'):
            # 실제로는 row별로 동작하는데 효율성 위해 미니배치로 처리
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)
            # i1 : 현재 배치 시작 index
            # i2 : 다음 배치 시작 index
            # count : 이번 배치에서 처리할 행의 개수
            # w : 미니배치에 해당하는 부분 W                 (count, self.columns)
            # Hinv : Hinv1(원본 H_inv)의 각행만큼의 복사본   (count, self.columns, self.columns)
            # mask : 현재 미니배치 내 가중치들에 대한 마스크  (count, self.columns)
            #                                               False 초기화 
            # rangecount : 0 ~ count-1  배치내 상대적   인덱스
            # idxcount   : i1 ~ i2-1   전체 가중치 실제 인덱스
            start = self.prepare_sparse(w, mask, Hinv, H)
            # start      : 미니배치에서 모든 행에 대해 이미 0 처리된 w 카운트
            outlier = .25 * (self.quantizer.scale ** 2)[i1:i2, :]
            scale = self.quantizer.scale[i1:i2, :]
            zero = self.quantizer.zero[i1:i2, :]

            tick = time.time()

            # --- 3. 반복적 가중치 양자화 루프 (논문의 Iterative Quantization) ---
            for quant_step in range(start, self.columns + 1):
                # ipdb.set_trace()
                # 현재 가중치 w를 양자화한 후보 q_candidate 계산
                q_candidate = quantize(w, scale, zero, self.quantizer.maxq) # quant.py의 함수
                # 양자화로 인한 제곱 오차 err 계산
                err = (w - q_candidate) ** 2
                # Hinv의 대각 성분 diag ([H^-1]_pp) 추출
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)

                # --- 2. 양자화할 가중치 선택 (논문 식 (7)의 argmin 부분) ---
                # OBS 점수 계산: err / diag
                scores = err / diag #  논문 수식 7 
                scores[mask] = float('inf') # 이미 처리된 가중치는 제외
                err[mask] = 0 # 오차도 0으로 (이미 처리됨)
                # 행별로 점수가 가장 낮은 가중치의 인덱스 j 선택
                j = torch.argmin(scores, 1)

                # --- 4. 이상치 처리 적용 (논문의 Outlier 처리 휴리스틱) ---
                sel = torch.any(err > outlier, 1) # 이상치 조건 만족 여부
                sel &= w[rangecount, j] != 0      # 선택된 가중치가 0이 아닌지 / 실제로 유의미한 오차를 발생시킬 수 있는, 0이 아닌 가중치에 대해서만 활성화되도록 보장
                if torch.any(sel):                # 이상치가 있다면
                    j[sel] = torch.argmax(err[sel, :], 1) # 해당 행에서는 오차가 가장 큰 가중치를 선택
                # 즉 j 는 기본적으로 행단위로 가장 score 가 작은 idx 이지만  outlier 가 있을 경우 (가장 큰 아웃라이어로 )우선적으로 선택된다. 
                
                # 손실 기록 및 실제 양자화
                Losses[i1:i2, quant_step] = scores[rangecount, j]
                q1 = q_candidate[rangecount, j] # 선택된 가중치 j에 대한 양자화 값
                Q[idxcount, j] = q1             # 최종 양자화 행렬에 저장
                # --- 2. 나머지 가중치 업데이트 (논문 식 (7)의 delta_p 부분) ---
                row = Hinv[rangecount, j, :] # H_:,p^-1 (선택된 가중치 j에 해당하는 Hinv의 행)   공식에 따르면 col 가져오는건데 왜 row 라고 이름 붙이는 지는 모르겠음.
                d = diag[rangecount, j]      # [H^-1]_pp (선택된 가중치 j에 해당하는 Hinv 대각 성분)
                # w_new = w_old - H_:,p^-1 * ( (w_p_old - q_p_quantized) / [H^-1]_pp )

                delta_p = -row * ((w[rangecount, j] - q1) / d).unsqueeze(1)
                w += delta_p # 나머지 가중치 업데이트  w 에 delta_p 를 더함   코드상으론 전체 w 를 다 업데이트 하긴 하는데 논리상,  이번 loop 에서  양자화된 가중치들의 index 는 영향이 없음.

                mask[rangecount, j] = True # 처리된 가중치 마스크

                if quant_step == self.columns: # 모든 열(가중치)이 처리되었으면 종료
                    break
                # --- OBS의 핵심: 역 헤시안 Hinv 업데이트 (Lemma 1 적용) ---
                # 다음 반복을 위해, 방금 "처리된" 가중치 j의 영향을 Hinv에서 제거
                row /= torch.sqrt(d).unsqueeze(1) # 스케일링된 Hinv 행
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1)) # Hinv 업데이트 . 이론적으로 Hinv1 가 대칭행렬 (실제로는 부동소수점 연산으로 인해 대칭이 약간 벗어나긴함)
                
            Losses[i1:i2, :] /= 2 # 최종 손실 조정

            torch.cuda.synchronize()
            # print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

        print('error', torch.sum(Losses).item())
        self.layer.weight.data = Q.reshape(self.layer.weight.shape)
        
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)

    def nmprune(self, n=2, m=4, parallel=32):
        W, H, Hinv1, Losses, perm = self.prepare(columnslast=True)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)

            buckets = torch.zeros((count, self.columns // m, 1), device=self.dev)

            tick = time.time()

            for zeros in range(1, self.columns + 1):
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = w ** 2 / diag 
                tmp = (buckets >= n).repeat((1, 1, m)).flatten(1)
                scores[mask | tmp] = float('inf')
                j = torch.argmin(scores, 1)
                Losses[i1:i2, zeros] = scores[rangecount, j]
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * (w[rangecount, j] / d).unsqueeze(1)
                mask[rangecount, j] = True
                buckets[rangecount, torch.div(j, m, rounding_mode='floor'), :] += 1
                if zeros == self.columns * n / m:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            Losses[i1:i2, :] /= 2
            w[mask] = 0
            W[i1:i2, :] = w

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

        print('error', torch.sum(Losses).item())
        W = W[:, torch.argsort(perm)]
        self.layer.weight.data = W.reshape(self.layer.weight.shape)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)

    def prepare_unstr(self, parallel=32):
        # 1. 기본 준비: 가중치(W), 헤시안(H), 역 헤시안(Hinv1), 손실 기록용 텐서(Losses) 준비
        # 이 Losses는 self.Losses로 저장되어 멤버 변수로 사용됨
        W, H, Hinv1, self.Losses = self.prepare() #

        # 2. 가중치 변화 추적을 위한 Traces 리스트 초기화
        self.Traces = [] #

        # 3. 행(row) 단위 병렬 처리를 위한 루프
        for i1 in range(0, self.rows, parallel):
            # 현재 미니배치 처리를 위한 변수들 준비 (w: 현재 행들의 가중치, Hinv: 복제된 역헤시안 등)
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1) #
            # 이미 0인 가중치가 있다면 Hinv 조정 및 OBS 시작점(start) 결정
            start = self.prepare_sparse(w, mask, Hinv, H) #

            # 4. 현재 미니배치에 대한 가중치 변화 추적(Trace)을 위한 텐서 초기화
            # Trace 텐서는 (제거된 가중치 수 + 1, 현재 미니배치 행 수, 전체 열 수) 크기를 가짐
            Trace = torch.zeros((self.columns + 1, count, self.columns), device=self.dev) #
            Trace[0, :, :] = w # 0개의 가중치가 제거된 상태 (원본 w)를 Trace의 첫 번째 슬라이스에 저장
            Trace[:start, :, :] = w # 이미 0인 가중치를 고려한 시작점(start) 이전까지는 원본 w와 동일하다고 간주

            tick = time.time()

            # 5. 반복적 가중치 제거 루프 (OBS 기반 비정형 가지치기)
            # 'start'는 이미 0인 가중치를 제외하고 실제 OBS 제거를 시작할 단계를 의미
            for zeros in range(start, self.columns + 1): # zeros는 제거할 가중치의 누적 개수를 의미
                # Hinv의 대각 성분 추출 ([H^-1]_pp)
                diag = torch.diagonal(Hinv, dim1=1, dim2=2) #
                # OBS 점수 계산: w_p^2 / [H^-1]_pp. 점수가 낮은 가중치가 제거 우선순위가 높음.
                scores = (w ** 2) / diag #
                scores[mask] = float('inf') # 이미 처리(제거)된 가중치는 무한대의 점수를 부여하여 다시 선택되지 않도록 함
                # 현재 처리 중인 각 행(row)에서 가장 낮은 점수를 가진 가중치의 인덱스(j)를 찾음
                j = torch.argmin(scores, 1) #
                # 선택된 가중치(j)를 제거했을 때의 손실(score)을 self.Losses에 기록
                self.Losses[i1:i2, zeros] = scores[rangecount, j] #

                # OBS 가중치 업데이트:
                # 선택된 가중치(j)를 0으로 만드는 것에 대한 오차를 보상하기 위해
                # 나머지 아직 제거되지 않은 가중치들을 업데이트.
                row = Hinv[rangecount, j, :] # H_:,p^-1 (선택된 가중치 j에 해당하는 Hinv의 행)
                d = diag[rangecount, j]      # [H^-1]_pp (선택된 가중치 j에 해당하는 Hinv 대각 성분)
                # w_new = w_old - H_:,p^-1 * ( w_p_old / [H^-1]_pp )
                w -= row * (w[rangecount, j] / d).unsqueeze(1) # 나머지 가중치 업데이트

                # 방금 제거한(0으로 만들) 가중치를 mask에 True로 표시
                mask[rangecount, j] = True #
                # 실제로 마스크된 위치의 가중치 값을 0으로 설정
                w[mask] = 0 #
                # 현재 단계(zeros개의 가중치가 제거된 상태)의 가중치 행렬 w를 Trace에 기록
                Trace[zeros, :, :] = w #

                # 모든 가중치가 제거되었다면 루프 종료
                if zeros == self.columns:
                    break

                # 역 헤시안 Hinv 업데이트 (Lemma 1 적용)
                row /= torch.sqrt(d).unsqueeze(1) # 스케일링된 Hinv 행
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1)) # Hinv 업데이트

            # 손실 값을 2로 나눔 (OBS 공식 관련 조정)
            self.Losses[i1:i2, :] /= 2 #
            # 계산된 Trace를 CPU 메모리로 옮겨 저장 (GPU 메모리 절약)
            self.Traces.append(Trace.cpu()) #

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

    def prune_unstr(self, sparsities):
        return self.prune_blocked(sparsities)

    def prepare_blocked(self, size=4, parallel=32):
        W, H, Hinv1, Losses, perm = self.prepare(columnslast=True)

        self.Traces = []
        blockcount = self.columns // size
        self.Losses = torch.zeros((self.rows, blockcount + 1), device=self.dev)
        rangeblockcount = torch.arange(blockcount, device=self.dev)
        rangecolumns = torch.arange(self.columns, device=self.dev)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, _, rangecount, _ = self.prepare_iter(i1, parallel, W, Hinv1)

            mask = torch.zeros((count, blockcount), device=self.dev).bool()
            mask1 = torch.zeros((count, blockcount, size), device=self.dev).bool()
            Trace = torch.zeros((blockcount + 1, count, self.columns), device=self.dev)
            Trace[0, :, :] = w
            rangeblockunroll = torch.arange(count * blockcount, device=self.dev)
            blockdiagidx = rangeblockcount.repeat(count)
            rangeunroll = torch.arange(self.columns * count, device=self.dev)
            diagidx = rangecolumns.repeat(count)
            paroffset = blockcount * rangecount
            expandrows = torch.arange(size, device=self.dev).unsqueeze(0).repeat(count, 1)
            expandrows += self.columns * rangecount.unsqueeze(1) 

            tick = time.time()

            for dropped in range(1, blockcount + 1):
                blocks = Hinv.reshape(count * blockcount, size, blockcount, size)
                blocks = blocks[rangeblockunroll, :, blockdiagidx, :]
                invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                w1 = w.reshape((count * blockcount, 1, size))
                lambd = torch.bmm(w1, invblocks)
                scores = torch.sum(lambd * w1, (1, 2))
                scores = scores.reshape((count, blockcount))
                scores[mask] = float('inf')
                j = torch.argmin(scores, 1)
                self.Losses[i1:i2, dropped] = scores[rangecount, j]

                tmp = (expandrows + size * j.unsqueeze(1)).flatten()
                rows = Hinv.reshape((-1, self.columns))[tmp]
                rows = rows.reshape((count, size, self.columns))
                tmp = paroffset + j
                d = invblocks[tmp]

                w -= torch.bmm(lambd[tmp], rows).squeeze(1)
                mask[rangecount, j] = True
                mask1[mask] = True
                tmp = mask1.flatten(1)
                w[mask1.flatten(1)] = 0
                Trace[dropped, :, :] = w

                if dropped == self.columns:
                    break
                Hinv -= torch.bmm(rows.transpose(1, 2), torch.bmm(d, rows))
                Hinv = Hinv.reshape((count * self.columns, self.columns))
                tmp = mask1.flatten()
                Hinv[rangeunroll[tmp], diagidx[tmp]] = 1
                Hinv = Hinv.reshape((count, self.columns, self.columns))
            self.Losses[i1:i2, :] /= 2
            Trace = Trace[:, :, torch.argsort(perm)]
            self.Traces.append(Trace.cpu())

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

    def prune_blocked(self, sparsities):
        parallel = self.Traces[0].shape[1]
        blockcount = self.Traces[0].shape[0] - 1
        losses = self.Losses[:, 1:].reshape(-1)
        order = torch.argsort(losses)
        Ws = [torch.zeros((self.rows, self.columns), device=self.dev) for _ in sparsities]
        losses = [0] * len(sparsities) 
        for i in range(self.rows):
            if i % parallel == 0:
                Trace = self.Traces[i // parallel].to(self.dev)
            for j, sparsity in enumerate(sparsities):
                count = int(math.ceil(self.rows * blockcount * sparsity))
                perrow = torch.sum(
                    torch.div(order[:count], blockcount, rounding_mode='trunc') == i
                ).item()
                losses[j] += torch.sum(self.Losses[i, :(perrow + 1)]).item()
                Ws[j][i, :] = Trace[perrow, i % parallel, :]
        for sparsity, loss in zip(sparsities, losses):
            print('%.4f error' % sparsity, loss)
            if DEBUG:
                tmp = self.layer.weight.data.clone()
                self.layer.weight.data = Ws[sparsities.index(sparsity)].reshape(self.layer.weight.shape) 
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)
                self.layer.weight.data = tmp
        return Ws

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
