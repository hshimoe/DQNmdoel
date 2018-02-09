# coding:utf-8
# [0]ライブラリのインポート
import numpy as np
import cv2
from Inv_Pendulum import InvertedPendulum, video


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
#ver 1.0
#参考URL:http://neuro-educator.com/rl2/
class DQN:        
	batch_size = 32
	gamma = 0.99
	def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=16):
		self.model = Sequential()
		self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
		self.model.add(Dense(hidden_size, activation='relu'))
		self.model.add(Dense(action_size, activation='linear'))
		self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
		self.model.compile(loss='mse', optimizer=self.optimizer)
		
	# 重みの学習
	def replay(self, memory, targetQN):
		inputs = np.zeros((self.batch_size, 4)) #入力データ(状態)
		targets = np.zeros((self.batch_size, 2))#出力データ(動作)
		mini_batch = memory.sample(self.batch_size) #ミニバッチサイズ
 
		for i, (state_b, action_b, reward_b, next_state_b, calc_reward) in enumerate(mini_batch):
			inputs[i:i + 1] = state_b #状態を渡す(転倒時は0)
			target = reward_b #報酬を渡す(常に1)

			if calc_reward :#オールゼロ(シミュレーション終了時は回避)
				# 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
				target_act = self.model.predict(next_state_b) 
				target = reward_b + self.gamma * np.max(target_act)
				
			targets[i] = self.model.predict(state_b)    # Qネットワークの出力
			targets[i][action_b] = target              # 教師信号
			targets = np.clip(targets, -1, 2000)	#2000step以上耐える必要はないため
		self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

		#print(targets)
			
 
class Memory:
	def __init__(self, max_size=10000):
		self.buffer = deque(maxlen=max_size)
 
	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
		return [self.buffer[ii] for ii in idx]

	def len(self):
		return len(self.buffer)



def bins(clip_min, clip_max, num):
	return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def get_action(state, timer, mainQN): #アクション取得
	explore_p = 0.01 + 0.99*np.exp(-0.0001*timer) #ランダム行動の発生確率
	
	if explore_p <= np.random.rand():#乱数の方が大きいならば,モデルを用いた行動
		Qs = mainQN.model.predict(state)[0]
		action = np.argmax(Qs)
	else:#乱数のほうが小さいならば,ランダムな行動
		action = np.random.choice([0, 1])
	
	return action

if __name__ == '__main__':
	ACTION_NUM=2 #行動の種類数
	first_pos = 0 #初期場所
	first_rad = np.pi/36#初期角度
	step_num = 200 #1step = 0.02[sec]/評価値にもなる 
	iter_num = 1000 #試行回数
	nearest_num = 10 #判定に用いる直近の数
	goal_reward = step_num * 0.95 #終了条件の報酬基準値
	total_reward_vec = np.zeros(nearest_num) #終了条件の報酬値
	mainQN = DQN() #DQNの初期化
	memory = Memory() #メモリ（データ保存先）の初期化
	reward = 1.#報酬は常に1
	timer = 0 #explore_p(乱数動作発生確率)の計算に用いるループ数
	
	for i in range(iter_num):
		pend = InvertedPendulum(first_pos, first_rad*np.random.uniform(-2, 2)) #倒立振子の初期化
		state = pend.get_state() #ステータスの取得
		trial_reward = 0
		for j in range(step_num):
			timer += 1 
			action = get_action(state, timer, mainQN) #アクションの取得
			pend.do_action(action) #アクションの更新
			next_state = pend.get_state()#ステータスの取得
			trial_reward += reward #1シミュレーションにおける報酬

			if pend.calc_reward():#終了判定
				reward = 1.
			else :
				reward = -1
				#next_state = np.zeros(state.shape)#終了した場合初期ステータス[0,0,0,0]を渡す?報酬系の代わりの処理
	
			memory.add((state, action, reward, next_state, pend.calc_reward()))#メモリに必要事項を渡す
			state = next_state #状態の更新

			if (memory.len() > 32) : #32以上になったら学習開始(ミニバッチのサイズの問題)
				mainQN.replay(memory, mainQN) #学習部分

			if j >= goal_reward or not(pend.calc_reward()): #報酬値が目標値以上若しくは転倒した場合
				print('%d Episode finished after %d time steps / mean %f' %(i, j + 1, total_reward_vec.mean()))	
				total_reward_vec[i % nearest_num] = trial_reward
				break

		if (total_reward_vec.mean() >= goal_reward): #学習終了判定
			print('Episode %d train agent successfuly!' % i)
			break


	#描画処理
	plt_plant = InvertedPendulum(first_pos, first_rad)
	x_history = [first_pos]
	angle_history = [first_rad]
	plt_action = get_action(plt_plant.get_state(), timer, mainQN)
	for i in range(step_num*10):
		plt_plant.do_action(plt_action)
		x_history.append(plt_plant.get_car_x())
		angle_history.append(plt_plant.get_car_theta())
		plt_state = np.reshape(plt_plant.get_state(), [1, 4])
		plt_action = get_action(plt_state, timer, mainQN)

	video(x_history, angle_history, plt_plant.l, plt_plant.t)
	

	cap = cv2.VideoCapture("tmp.mp4")
	# 動画終了まで繰り返し
	while(cap.isOpened()):
	# フレームを取得
		ret, frame = cap.read()
	# フレームを表示
		if frame is None:
			break
		cv2.imshow("Flame", frame)
		cv2.waitKey(1)

	cap.release()
	cv2.destroyAllWindows()

