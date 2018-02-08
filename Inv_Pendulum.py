import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pynput.keyboard import Key, Listener

class InvertedPendulum(object):
	#定数項
	actions = [0, 1, 2] #0=左 1=動かさない 2=右
	g = 9.8 #重力加速度[m/s^2]
	M = 1 #台車の質量[kg]
	m = .1 #振り子の先の重さ[kg]
	l = 0.5 #振り子の長さ(全長の半分)[m]
	t = 0.02 #描画 time step[sec]
	t_num = 500 #終了までのステップ数[回]
	t_one = t / t_num #タイムステップとして利用されてる
	

	def __init__(self, x, theta, noisy=False):
		self.theta = theta #角度[rad]
		self.theta_dot = 0 #角速度[rad]
		self.x = x #台車の位置[m]
		self.x_dot = 0#台車の速度
		self.u  = 0 #加える力[N]
		self.noisy = noisy #動作に対する誤差発生の有無[boolean]
	#基本ここを操作
	def do_action(self, a):
		assert a in self.actions, str(a)+" is not in actions"

		if a == 0:
			self.u = -10.#左
		elif a == 2:
			self.u = 0.	#停止
		elif a == 1:
			self.u = 10. #右
	
		if self.noisy: #微細動作
			self.u += np.random.uniform(-10, 10)
		self.update_state()

	def update_state(self): #0.1秒用のシミュレーション
		for i in range(self.t_num):
			sin_theta = np.sin(self.theta)
			cos_theta = np.cos(self.theta)
			ml = self.m * self.l
			total_M = self.m + self.M

			temp = (self.u + ml * self.theta_dot**2 * sin_theta) / total_M
			theta_acc = ((self.g * sin_theta - cos_theta * temp) / (self.l * (4/3 - self.m * cos_theta**2 / total_M))) #角加速度
			x_acc = temp - ml * theta_acc * cos_theta / total_M #台車加速度

			self.theta_dot += self.t_one * theta_acc
			self.theta += self.t_one * self.theta_dot
			self.x_dot += self.t_one * x_acc
			self.x += self.t_one * self.x_dot

	def calc_reward(self):
		if -np.pi/6 <= self.theta <= np.pi/6 \
			and -2. <= self.x <= 2.\
			and -2. <= self.x_dot <= 2.\
			and -7. <= self.theta_dot <= 7:
			return True
		else:
			return False

	def complexly_reward(self):
		if -np.pi/6 <= self.theta <= np.pi/6 \
			and -2. <= self.x <= 2.\
			and -2. <= self.x_dot <= 2.\
			and -7. <= self.theta_dot <= 7:
			reward = (2-np.abs(self.x)+np.pi/6-np.abs(self.theta))/(2+np.pi/6)
			return reward
		else:
			return -self.t_num
	def get_state(self):
		return np.reshape((self.x, self.x_dot, self.theta, self.theta_dot),[1, 4])

	def get_car_x(self):
		return self.x

	def get_car_theta(self):
		return self.theta



def video(x_history, angle_history, l, t):
	#描画初期設定
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10,10), ylim=(-2, 2))
	ax.grid()
	line, = ax.plot([], [], 'o-', lw=2)
	time_text = ax.text(0.02, 0.95, 'aaaaaa', transform=ax.transAxes)

	def init():
		line.set_data([], [])
		time_text.set_text('')

	def animate(i):
		line.set_data([x_history[i], x_history[i]+2*l*np.sin(angle_history[i])],
					  [0, 2*l*np.cos(angle_history[i])])
		time_text.set_text('time = {0:.1f}'.format(i*t))
		return line, time_text
	
	ani = animation.FuncAnimation(fig, animate, frames=range(len(x_history)),interval=1000*t, blit=False, init_func=init)
	plt.show()
	#ani.save('tmp.mp4')


def video_2(pend, l, t):
	#描画初期設定
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2,2), ylim=(-2, 2))
	ax.grid()
	line, = ax.plot([], [], 'o-', lw=2)
	time_text = ax.text(0.02, 0.95, 'aaaaaa', transform=ax.transAxes)

	key_flag = 2
	action_num = 2

	def init():
		line.set_data([], [])
		time_text.set_text('')
	def on_press(key):
		nonlocal key_flag 
		key_flag = 1
		print("right")
	def on_release(key):
		nonlocal key_flag
		key_flag =0
		print("left")
	
	def animate(i):	
		nonlocal action_num
		nonlocal key_flag
		if key_flag == 1:
			action_num = 1
		elif key_flag == 0:
			action_num = 0
		print(action_num)
		pend.do_action(action_num)
		line.set_data([pend.get_car_x(), pend.get_car_x()+2*l*np.sin(pend.get_car_theta())],
					  [0, 2*l*np.cos(pend.get_car_theta())])
		time_text.set_text('time = {0:.1f}'.format(i*t))
		return line, time_text



	listener = Listener(on_press=on_press, on_release=on_release)
	listener.start()
	try:
		ani = animation.FuncAnimation(fig, animate, frames=range(len(x_history)),interval=1000*t, blit=False, init_func=init)
	except KeyboardInterrupt:
		listener.terminate()
	plt.show()


if __name__ == '__main__':
	first_rad = np.pi/360
	plant = InvertedPendulum(0, first_rad)
	angle_history = [first_rad]
	x_history = [0.]
	video_2(plant, plant.l, plant.t)

