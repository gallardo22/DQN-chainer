#-*- coding: utf-8 -*-
#ver0.41 2016/11/22
#python全体で使うような奴の追加
import argparse					#起動時の引数設定
import os						#ファイルの削除
import sys						#シス
import copy						#データコピーテム関連(ここではプログラムの強制終了とか)
import time						#時間取得
import random					#ランダム
import ConfigParser				#iniファイルいじる
import threading				#マルチスレッド
import csv						#csvファイルの扱い(出力用)
from collections import deque	#デック(deque)を扱えるようにする
from tqdm import tqdm			#プログレスバー
#ニューラルネットワークに関する追加
import numpy as np				#数値計算補助(CPU)
import cupy as cp				#数値計算補助(GPU)
import chainer					#ディープネットワークのフレームワーク
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, Variable
#画面操作に関する追加
import pyautogui as ag			#ディスプレイ内の画像検索、キーボード操作など
from PIL import Image			#RGB調べる
#from PyQt4.QtGui import QPixmap,QApplication
#from PyQt5.QtWidgets import QApplication, QWidget, QLabel
#from PyQt5.QtGui import *
from PyQt4.QtGui import *




#引数設定
parser = argparse.ArgumentParser(description='Learning game')
#行動選択の間隔(ms)
parser.add_argument('--interval', default=1000, type=int,
				help='interval of capturing (ms)')
#ゲームを回す回数
parser.add_argument('--play_number','-p' , default=1000, type=int,
				help='play number of count')
#学習方法(初期設定DQN)(まだ使ってない)
#Q・・・Q-learning
#PS・・・Profit Sharing
parser.add_argument('--learning_type','-l' , default='Q', type=str,
				help='learning type')
args = parser.parse_args()

#起動時の時間を保存
start_time = time.time()
#サーチ間隔(ms)
interval = args.interval / 1000.0
#gpuがあればcupyなければnumpyを使う(cpu未確認)
xp = cuda.cupy if cuda.available else np
#ランダムシードの決定
np.random.seed(0)
#pyautoguiのコード実行後の待ち時間(s)
ag.PAUSE = 0
#画面の位置設定
#キャプチャするウィンドウの取得
app = QApplication(sys.argv)
window_id = app.desktop().winId()
#ゲーム画面(学習対象)の位置(左上からの)とサイズ
learning_left = 0
learning_top = 0
learning_width = 800
learning_height = 600
#学習画面の縮小度(xピクセルに一つの値を使用)
interval_w = 2
interval_h = 2


#メイン
def main():
	print (chainer.__version__)
	#ゲーム画面認識してあったら画面にフォーカス移動,なければ終了
	search = Thread_search('./image/start.png', 5)
	#スレッドスタート
	search.start()
	#ここでは他に動かすものがないので終わるまで待つ(この先も使うことはなかった)
	search.join()
	#値が返って来ていてディスプレイ内に認識したい画面が全部入っていたら
	if search.left != -1 and ag.onScreen(search.left + learning_left + learning_width, search.top + learning_top +learning_height):
		#画面内の適当な位置にカーソルを持って行きクリック
		ag.moveTo(search.left+48, search.top+16)
		ag.click()
		#画面外にカーソルを外しておく
		time.sleep(1)
		ag.press('t')
		time.sleep(1)
		#迷路の生成
		time.sleep(1)
		ag.press('r')
		time.sleep(1)
		#迷路の写真取る
		ag.press('m')
		time.sleep(1)
		ag.press('g')
		time.sleep(1)
		ag.press('m')
		ag.moveTo(search.left-48, search.top-16)
	#ダメならエラーはいて終了
	else :
		print '\033[31mError01:認識失敗　ディスプレイ内に画面がすべて入っていない場合があります\033[0m'#error01
		sys.exit()

	#学習したい画面左上の座標を保存しておく(状態取得に使う)
	left = search.left + learning_left
	top = search.top + learning_top
	#エージェント作成
	agent = Agent(learning_width, learning_height, interval_w, interval_h)
	#調整エージェント作成
	tmp_agent = Agent(learning_width, learning_height, interval_w, interval_h)
	#インターバル調整
	
	

	#ゲームを回す回数を決めておきその中かを確認
	for i in range(1,args.play_number+1) :
		#100回に1度インターバル調整(うまくいってない？)
		#if i % 100 == 1:
			#インターバル調整
		#	interval = interval_adjust(tmp_agent,window_id)
		#最後に保存したスコアとゲームの状態と行動価値の最大値を初期化
		game_state = 2
		print '%d回目のゲーム開始'%i
		#get_reward内でgame_state更新してるからこっちではそのまま調べる
		while game_state < 3 and agent.step < 6000:
			#今の時間をとっておく
			now_time = time.time()
			#現在の状態を取得
			state = get_state(window_id, left, top, learning_width, learning_height, interval_w, interval_h, agent)
			action_state = set_state(agent, state, learning_width, learning_height, interval_w, interval_h)
			#取得した状態に応じて行動を選択
			action_choice, q = agent.get_action(action_state)
			fw = open('Q_0301_original.csv','a')
			fw.write('%d回目,%f,%d\n'% (i, q, action_choice))
			fw.close()
			#選択された行動を実際に実行する
			set_action(action_choice)
			#報酬取得
			reward, game_state, ep_end, g = get_reward()
			time.sleep(0.2)
			#一時メモリに保存
			agent.stock_tmp(state, action_choice, reward, ep_end)
			#経験メモリに入れるだけの情報が集まったら
			if len(agent.tmp_memory) > agent.tmp_size:
				#経験をストックする
				agent.stock_experience(g)
			#ステップを１増やす
			agent.step += 1
			agent.reduce_epsilon()
			#最初の状態取得から次の取得の間の時間
			interval_time = time.time()-now_time
			#情報表示
			print '%d回目\tstep%d\ttime:%01.4fs\taction:%d\tQ:%01.4f\tepsilon:%01.4f'% (i,agent.step, time.time()-now_time, action_choice, q, agent.epsilon)
			agent.total_step += 1
		print '%d回目のゲーム終了'%i
		print 'ここまでのステップ数%d'%agent.total_step
		f = open('keika_0301_original.csv','a')
		f.write('%d,%01.4f\n'% (agent.step, agent.epsilon))
		f.close()
	
		#開始からの時間を表示
		time_tmp = time.time() - start_time
		print '実行時間 : %02d:%02d:%02.2f' % (time_tmp//3600, time_tmp%3600//60, time_tmp%60)
		#学習をゲーム終了時にする
		agent.train(i,state)

		agent.step = 0
		agent.goal += 1
		
		time.sleep(0.5)
		ag.press('i')
		time.sleep(0.5)


#-----------------↓関数とクラス↓--------------------

#特定画像がディスプレイ内に存在するか検索をするマルチスレッド用クラス
#マルチスレッドにして使うときは来なかったから意味はない
#pic・・・探したい画像パス
#timeout・・・タイムアウトまでの時間(秒)0のときタイムアウトを行わない
#スレッド実行後使う値・・・見つけた座標x,y。タイムアウトもしくは見つかるまで(-1,-1)
class Thread_search(threading.Thread):
	def __init__(self, pic, timeout):
		threading.Thread.__init__(self)
		self.pic = pic
		self.timeout = timeout
		#変数の初期化(特に必要なのはleft,top)
		self.left, self.top, self.width, self.height = [-1, -1, -1, -1]

	#startで実行される部分
	def run(self):
		#探し始めた時間を記録
		search_start_time = time.time()
		while True :
			#画像と同じ部分を探す(１箇所のみ)
			pos_search = ag.locateOnScreen(self.pic)
			#取得できた時
			if pos_search is not None:
				#取得した位置を変数に入れる
				self.left, self.top, self.width, self.height = pos_search
				print '取得に%01.5fsかかりました'% (time.time()-search_start_time)
				#座標をかえせる状態になったからbreak
				break
			#取得できなかった時
			else :
				#タイムアウトするかどうかの確認
				if (time.time()-search_start_time) > self.timeout and self.timeout != 0:
					print '%dsで取得できませんでした'% self.timeout
					print 'タイムアウト'
					break
				#タイムアウトの時間でないときwhileにもどる

#行動選択の数値を3種の動作に分けてからiniに保存
def set_action(action):
	#choice(0~2)をもとに3つの行動に分ける
	#0が前進、１が左に向く、２が右に向く
	
	if action == 0:
		ag.keyDown('w')
		time.sleep(0.1)
		ag.keyUp('w')

	elif action == 1:
		ag.keyDown('a')
		time.sleep(0.1)
		ag.keyUp('a')

	elif action == 2:
		ag.keyDown('d')
		time.sleep(0.1)
		ag.keyUp('d')


#状態state(画面のRGB情報)を入手
#win_id・・・みるウィンドウ
#left・・・学習する画面の左上のｘ座標
#top・・・学習する画面の左上のｘ座標
#width・・・学習する画面の横幅
#height・・・学習する画面の縦幅
#interval_w・・・横方向の情報入手間隔
#interval_h・・・縦方向の情報入手間隔
def get_state(win_id, left, top, width, height, interval_w, interval_h, agent):
	#pixmapにウィンドウの情報を渡す
	pixmap = QPixmap.grabWindow(win_id, left, top, width, height)
	#QImageへの変換
	image = pixmap.toImage()
	bits= image.bits()
	bits.setsize(image.byteCount())
	#数ピクセルに一つの割合でImage型RGBモードで保存しなおす
	screen = Image.fromarray(np.array(bits).reshape((height, width, 4))[::interval_w,::interval_h,2::-1])
	state = np.asarray(screen)

	##いらない部分(画像保存できるか試してみる)
	#png2 = './piclog/' + str(agent.step) + '.png'
	#保存
	#screen.save(png2)
	
	#書き込み可能に
	state.flags.writeable = False
	#元画像の緑だけを取り出す
	state = state[:,:,1]
	#戻り値にstate(現在の状態)を返す	
	return state

#行動選択時に渡す入力を作成する
def set_state(agent, state, learning_width, learning_height, interval_w, interval_h):
	#返り値用のリスト作成
	tmp = []
	#まず持ってるだけ一時メモリの情報をもらう
	for i in range(min(agent.tmp_size-1, len(agent.tmp_memory))):
		tmp.append(agent.tmp_memory[i][0])
	#今の状態を追加
	tmp.append(state)
	#足りない分0行列を付け足す
	while len(tmp) < agent.tmp_size:
		tmp.insert(0, state)#np.zeros( [learning_width//interval_w, learning_height//interval_h] ))
	return tmp

#報酬の決定
#-1から1にクリッピングする
#報酬とゲームの状態と今のスコアを返す
def get_reward():
	goal = ag.locateOnScreen('./image/GOAL.png')
		#if goal == None and np.array_equal(stat,sd) == True :
		#	return -1,2
	if goal == None:
		return 0,2,0,goal
	else :
		return 1,3,1,goal
			
#インターバルの時間を調整する(インターバル以外の動作を行い時間を測る)



#ディープラーニングを行うためのニューラルネットクラス[1]
class Neuralnet(chainer.Chain):

	def __init__(self, in_size):
		#モデル(ネットワークの形)の定義
		super(Neuralnet, self).__init__(
            #チャネル数K,出力チャネル数M,フィルタサイズH,padパディング[2]
            conv1=L.Convolution2D(in_size, 16, 8,stride=4,pad=2),#数値は適当(どうやって決めるんだ？)
            conv2=L.Convolution2D(16, 32, 4, pad=2),
            conv3=L.Convolution2D(32, 64, 2, pad=1),
            #教えてくれた値,行動種類
			#fc4=L.Linear(126,128),	　＃アウトオブメモリーなので抜いた
            fc5=L.Linear(502656,3, initialW=np.zeros((3, 502656), dtype=np.float32))
        )
		#gpuに持ってくところ
		if cuda.available:
			cuda.get_device(0).use()
			self.to_gpu()

	#x・・・入力
	def __call__(self, x):
		h = F.relu(self.conv1(x))#入力のスケールを[0,1]に
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		#h = F.relu(self.fc4(h))
		y = self.fc5(h)
		return y

#学習を行うエージェント[3]
class Agent():
	
	def __init__(self,learning_width, learning_height, interval_w, interval_h):
		#経験メモリ
		self.tmp_memory = deque()
		self.tmp_size = 1#一つの状態が持つ画面数(実際のメモリサイズは+1)
		self.memory = deque()#メモリ(キュー)
		self.batch_size = 8#バッチサイズ
		#モデル設定
		self.model = Neuralnet(self.tmp_size)
		self.target_model = copy.deepcopy(self.model)
		#最適化設定
		self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
		self.optimizer.setup(self.model)
		#学習に使う画面サイズ(上のをコピーしてるだけ)
		self.learning_width = learning_width
		self.learning_height = learning_height
		self.interval_w = interval_w
		self.interval_h = interval_h

		#εグリーディに使用する値
		self.epsilon = 1#初期値
		self.epsilon_decay = 1.0/200000#一回に下げる値
		self.epsilon_min = 0.05#最低値
		self.exploration = 70
		#その他設定
		self.step = 0#ステップ数	
		self.total_step = 0
		self.goal = 0
		self.gamma = 0.90# 割引率(ガンマ)


	#一時的に状態などを保存しておく。
	#画面の情報が必要数集まってから経験メモリに入れるため
	def stock_tmp(self, state, action, reward, ep_end):
		self.tmp_memory.append((state, action, reward, ep_end))
		#メモリの長さ設定を超えていたら後ろを消す
		if len(self.tmp_memory) > self.tmp_size+1:
			self.tmp_memory.popleft()

	#経験を貯める(設定メモリ上限を超えたら後ろから消す)
	#引数は全部メモリに入れる内容(状態,行動,報酬,次の状態,例外)
	def stock_experience(self,g):
		state = []
		state_dash = []
		#stateには1〜4つ目、state_dashには2〜5つ目のデータを入れる
		for i in range(self.tmp_size):
			state.append(self.tmp_memory[i][0])
			state_dash.append(self.tmp_memory[i+1][0])
		#ほかは4つ目のデータを入れる
		action = self.tmp_memory[self.tmp_size-1][1]

		if g == None:
			reward = self.tmp_memory[self.tmp_size-1][2]
			ep_end = self.tmp_memory[self.tmp_size-1][3]
		else :
			reward = self.tmp_memory[self.tmp_size][2]
			ep_end = self.tmp_memory[self.tmp_size][3]
		self.memory.append((state, action, reward, state_dash, ep_end))

	#ミニバッチを作れるサイズまでメモリを無で埋める
	def fill_memory(self, sta):
		#バッチサイズで割れない時
		while len(self.memory) % self.batch_size != 0:
			#中身が全部ゼロの状態作成
			zero_state = sta
			state = []
			state_dash = []
			#全部０データ入れる
			for i in range(self.tmp_size):
				state.append(zero_state)
				state_dash.append(zero_state)
			#例外設定することで学習しない
			action = 0
			reward = 0
			ep_end = 1
			self.memory.append((state, action, reward, state_dash, ep_end))

	#Q学習による順伝搬
	def Q_forward(self, state, action, reward, state_dash, ep_end):
		#状態をVariable型に入れる
		s = Variable(state)
		s_dash = Variable(state_dash)
		#順伝搬
		Q = self.model(s)
		tmp = self.target_model(s_dash)#次の行動の方はtarget_modelを使用
		#次の行動のQの最大値求める奴
		tmp = list(map(np.max, tmp.data))
		max_Q_dash = xp.asanyarray(tmp, dtype=np.float32)
		target = xp.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
		#この後の計算でgpuモードが使えないので一度cpuに移動
		max_Q_cpu = cuda.to_cpu(max_Q_dash)		
		target_cpu = cuda.to_cpu(target)
		for i in xrange(self.batch_size):
			#Q値計算(エピソードが終了するところでは報酬のみ)
			target_cpu[i, action[i]] = reward[i] + (self.gamma * max_Q_cpu[i]) * (not ep_end[i])
			#print(target_cpu[i,action[i]])
			
		#またネットワーク使うのでgpuに戻す
		target2 = xp.array(target_cpu)
		#printtmp = (Q.data - Variable(target2).data)##
		#print  printtmp##
		#ロス計算
		loss = F.mean_squared_error(Q, Variable(target2))
		self.loss = loss.data
		return loss

	#メモリの中をランダムシャッフル(果たしているのか)
	def suffle_memory(self):
		mem = np.array(self.memory)
		return np.random.permutation(mem)

	#取り出したミニバッチのデータを取り出す
	def parse_batch(self, batch):
		#返すデータ用の配列作成
		state, action, reward, state_dash, ep_end = [], [], [], [], []
		#それぞれの配列にデータを入れていく
		for i in xrange(self.batch_size):
			state.append(batch[i][0])
			action.append(batch[i][1])
			reward.append(batch[i][2])
			state_dash.append(batch[i][3])
			ep_end.append(batch[i][4])
		#型をそれぞれにあったものを選択
		state = xp.asarray(state, dtype=np.float32)
		action = xp.asarray(action, dtype=np.int8)
		reward = xp.asarray(reward, dtype=np.float32)
		state_dash = xp.asarray(state_dash, dtype=np.float32)
		ep_end = xp.asarray(ep_end, dtype=np.bool)
		return state, action, reward, state_dash, ep_end

	#Experience Replayによるバッチ学習
	def experience_replay(self):
		#メモリをシャッフル
		mem = self.suffle_memory()
		perm = np.array(xrange(len(mem)))
		#メモリをバッチサイズごとに切ってそれぞれについて学習
		for start in tqdm(perm[::self.batch_size]):
			
			index = perm[start:start+self.batch_size]
			batch = mem[index]
			state, action, reward, state_dash, ep_end = self.parse_batch(batch)
			#勾配の初期化
			self.model.zerograds()
			#順伝搬・逆伝搬の実行
			loss = self.Q_forward(state, action, reward, state_dash, ep_end)
			loss.backward()
			#最適化の実行
			self.optimizer.update()
	
	#行動選択をする(εグリーディ)
	def get_action(self, state):
		sta = [state]
		sta = xp.asarray(sta, dtype=np.float32)
		s = Variable(sta)
		Q = self.model(s)			
		Q = Q.data[0]
		print Q
		#εの確率でランダムな行動をする(ランダムの時の行動価値は0)
		if np.random.rand() < self.epsilon:
			#18は行動の最大数(3*3*2)
			return np.random.randint(0, 3), 0
		#行動価値が一番高い行動とその値を返す
		else:
			#stateの配列を学習時に合わせる
			#sta = [state]
			#sta = xp.asarray(sta, dtype=np.float32)
			#s = Variable(sta)
			#Q = self.model(s)
			#Q = Q.data[0]
			a = np.argmax(Q)
			return xp.asarray(a, dtype=np.int8), max(Q)
	#行動選択のεの値更新(下がるだけ)
	def reduce_epsilon(self):
		#最小値より多くて設定時間を超えていれば
		if self.epsilon > self.epsilon_min and self.exploration < self.goal:	
			self.epsilon -= self.epsilon_decay




	#エピソード終了時に学習する用(エピソード中にやったら時間かかってしまったので)
	def train(self, i, state):
		self.fill_memory(state)
		self.experience_replay()
		if i % 5 == 0: #モデルの更新
			self.target_model = copy.deepcopy(self.model)
		#キューのリセット
		self.tmp_memory = deque()
		self.memory = deque()#メモリ(キュー)

#メイン(mainの中身を上に持っていくため)
if __name__ == '__main__':
	main()

#以下参考
#[1]DQNもどきの人のプログラム(https://github.com/trtd56/ClassicControl)
#[2]Chainerの構造(http://jprogramer.com/ai/3758)

