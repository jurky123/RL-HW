from agent import MahjongGBAgent
from gym.spaces import Dict, Discrete, Box
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class FeatureAgent(MahjongGBAgent):
    # 核心维度定义：采用 174 层专家级特征
    OBS_SIZE = 174
    ACT_SIZE = 235

    observation_space = Dict({
        'observation': Box(0, 1, (OBS_SIZE, 4, 9)), 
        'action_mask': Box(0, 1, (ACT_SIZE,))
    })
    action_space = Discrete(ACT_SIZE)
    
    # 174层特征偏移量定义
    OFFSET_OBS = {
        'SEAT_WIND' : 0,      # 1
        'PREVALENT_WIND' : 1, # 1
        'HAND' : 2,           # 4
        'PACK' : 6,           # 4*4*4 = 64
        'DISCARD' : 70,       # 4*25 = 100
        'UNSHOWN' : 170       # 4
    }
    OFFSET_ACT = {
        'Pass' : 0, 'Hu' : 1, 'Play' : 2, 'Chi' : 36,
        'Peng' : 99, 'Gang' : 133, 'AnGang' : 167, 'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}
    OFFSET_TILE['PUBLIC'] = 34    # 副露标志：公开
    OFFSET_TILE['CONCEALED'] = 35 # 副露标志：暗杠

    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for _ in range(4)]
        self.history = [[] for _ in range(4)]
        self.tileWall = [21] * 4 
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.hand = []
        self.prevalentWind = 0
        self.valid = []
        
        # 初始化 174x36 矩阵，最后 reshape (174, 4, 9)
        self.obs = np.zeros((self.OBS_SIZE, 36))
        # 写入初始门风
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1

    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()

        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = (self.tileWall[1] == 0)
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            
            for tile_in_hand in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile_in_hand])
                if self.hand.count(tile_in_hand) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile_in_hand])
            
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile_p, offer in self.packs[0]:
                    if packType == 'PENG' and tile_p in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile_p])
            return self._obs()

        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = (self.tileWall[(p + 1) % 4] == 0)
            return
        if t[2] == 'Invalid' or t[2] == 'Hu':
            self.valid = []
            return self._obs()

        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            self._history_embedding_append(p)
            
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    # 吃牌合法性检查
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()

        if t[2] == 'Chi':
            tile = t[3] # 吃的中间牌
            color = tile[0]; num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self._pack_embedding_append(p)
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2): self.shownTiles[color + str(num + i)] += 1
            if p == 0:
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2): self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tt in set(self.hand): self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tt])
                return self._obs()

        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self._pack_embedding_append(p)
            self.shownTiles[self.curTile] += 2
            if p == 0:
                self.valid = []
                for _ in range(2): self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tt in set(self.hand): self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tt])
                return self._obs()

        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self._pack_embedding_append(p)
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for _ in range(3): self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return

        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            self._pack_embedding_append(p)
            if p == 0:
                self.isAboutKong = True
                for _ in range(4): self.hand.remove(t[3])
                self._hand_embedding_update()
            return

        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    offset = self.OFFSET_OBS['PACK'] + 16 * p + i * 4 + 3
                    self.obs[offset, self.OFFSET_TILE[tile]] = 1
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
            else:
                self.valid = []
                if self._check_mahjong(tile, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()

    # --- 辅助方法 ---

    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid: mask[a] = 1
        self._unshown_embedding_update()
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + 4] = 0
        d = defaultdict(int)
        for tile in self.hand: d[tile] += 1
        for tile, count in d.items():
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + count, self.OFFSET_TILE[tile]] = 1

    def _pack_embedding_append(self, p):
        l = len(self.packs[p]) - 1
        packType, tile, offer = self.packs[p][-1]
        offset = self.OFFSET_OBS['PACK'] + 16 * p + l * 4
        if packType == 'CHI':
            color = tile[0]; num = int(tile[1])
            for i in range(-1, 2): self.obs[offset, self.OFFSET_TILE[color + str(num + i)]] = 1
            self.obs[offset : offset + 4, self.OFFSET_TILE['PUBLIC']] = 1
        elif packType == 'PENG':
            self.obs[offset : offset + 3, self.OFFSET_TILE[tile]] = 1
            self.obs[offset : offset + 4, self.OFFSET_TILE['PUBLIC']] = 1
        else: # GANG
            self.obs[offset : offset + 4, self.OFFSET_TILE[tile]] = 1
            self.obs[offset : offset + 4, self.OFFSET_TILE['PUBLIC' if offer or p==0 else 'CONCEALED']] = 1

    def _history_embedding_append(self, p):
        tile = self.history[p][-1]
        idx = min(len(self.history[p]) - 1, 24)
        self.obs[self.OFFSET_OBS['DISCARD'] + p * 25 + idx, self.OFFSET_TILE[tile]] = 1

    def _unshown_embedding_update(self):
        self.obs[self.OFFSET_OBS['UNSHOWN'] :, :] = 0
        local_shown = self.shownTiles.copy()
        for t in self.hand: local_shown[t] += 1
        for tile in self.TILE_LIST:
            uncount = 4 - local_shown[tile]
            if uncount > 0:
                self.obs[self.OFFSET_OBS['UNSHOWN'] : self.OFFSET_OBS['UNSHOWN'] + uncount, self.OFFSET_TILE[tile]] = 1

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = self.shownTiles[winTile] >= 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = sum(f[0] * f[1] for f in fans)
            return fanCnt >= 8
        except: return False

    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']: return 'Pass'
        if action < self.OFFSET_ACT['Play']: return 'Hu'
        if action < self.OFFSET_ACT['Chi']: return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']: return 'Peng'
        if action < self.OFFSET_ACT['AnGang']: return 'Gang'
        if action < self.OFFSET_ACT['BuGang']: return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']