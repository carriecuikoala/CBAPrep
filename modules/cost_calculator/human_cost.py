import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class CostVisualizer:
    @staticmethod
    def plot_complexity_components(components):
        """复杂度组成分解图"""
        plt.figure(figsize=(12, 8))

        # 生成任务标签（Task 1, Task 2,...）
        task_labels = ['data_extraction','data_labeling','error_correction']

        # 生成组件标签
        component_labels = ['Dimension', 'Samples', 'Uncertainty']

        # 创建DataFrame提高可读性
        df = pd.DataFrame(
            components,
            index=task_labels,
            columns=component_labels
        )

        # 绘制热力图
        ax = sns.heatmap(
            df.T,
            annot=True,
            fmt=".2f",
            cmap='Blues',
            cbar_kws={'label': 'Component Weight'}
        )

        # 优化标签
        ax.set_title("Task Complexity Component Analysis", pad=20, fontsize=14)
        ax.set_xlabel("Task Cases", fontsize=12)
        ax.set_ylabel("Complexity Components", fontsize=12)
        ax.figure.tight_layout()

        plt.show()


class HumanCostCalculator:
    def __init__(self ):
        """
        人工成本计算器初始化配置
        """
        # 时薪标准配置（单位：元/小时）
        # self.wage_levels = {
        #     'junior': 50,  # 初级人员
        #     'senior': 100,  # 中级人员
        #     'expert': 200  # 专家级
        # }
        # self.efficiency_levels = {
        #     'junior': 50,  # 初级人员
        #     'senior': 100,  # 中级人员
        #     'expert': 200  # 专家级
        # }

        self.levels = {
            'junior': {
                'wage':15,
                'time':2,
                'threshold':5,
                'argument':1,
                'number':80,
                'e_cost':10
            },  # 初级人员
            'senior': {
                'wage':30,
                'time':1.5,
                'threshold':7.0,
                'argument': 0.7,
                'number': 20,
                'e_cost': 50
            },  # 中级人员
            'expert': {
                'wage':300,
                'time':1,
                'threshold': None,
                'argument': 0.5,
                'number': 3,
                'e_cost': 500
            }
        }
        # # 复杂度阈值配置（用于动态选择人员级别）
        # self.complexity_thresholds = {
        #     'low': 2.0,  # 初级人员处理阈值
        #     'medium': 4.0  # 中级人员处理阈值
        # }

        # 基础时间配置（单位：小时）
        self.base_time = {
            'manual_labeling': 0.005,
            'manual_selecting': 0.5,
            # 'missing_value': 6,
            # 'outlier_detection': 10
        }

        # 经验调节系数
        self.gamma = 1.8
        self.complexity = None
        self.level = None
        self.quantity = 1000




    def calculate_complexity(self, d,d_max,n,n_max, mu):
        """
        计算任务复杂度指标θ
        :param n_max:任务处理样本量最大值
        :param d_max:任务处理维度最大值
        :param d: 数据维度
        :param n: 样本量
        :param mu: 模糊度 (0-1)
        :return: 复杂度θ
        """
        if d <= 0 or n <= 0:
            raise ValueError("数据维度和样本量必须大于0")
        if not 0 <= mu <= 1:
            raise ValueError("模糊度必须在0-1之间")

        self.complexity =10*(0.3 * d/d_max+ 0.1 * math.log(n)/math.log(n_max) + 0.6 * mu)

    def get_level(self):
        """
        根据复杂度选择人员级别
        :param
        complexity 复杂度指标
        :return: 时薪级别
        """
        if self.complexity < self.levels['junior']['threshold']:
            self.level = 'junior'
            # self.q['junior'] = random.randint(0.8*q_sum, q_sum)
            # self.q['senior'] = q_sum - self.q['junior']
        elif self.complexity < self.levels['senior']['threshold']:
            self.level = 'senior'
            # self.q['senior'] = random.randint(0.8*q_sum, q_sum)
            # self.q['expert'] = q_sum - self.q['senior']
        else:
            self.level = 'expert'
            # self.q['expert'] = q_sum


    def calculate_task_cost(self, task_type, d,d_max , n, n_max,mu):
        """
        计算单个任务的人工成本
        :param n_max:
        :param d_max:
        :param task_type: 任务类型
        :param d: 数据维度
        :param n: 样本量
        :param mu: 模糊度
        :return: 人工成本（元）
        """
        # 1. 获取基础时间
        t_base = self.base_time.get(task_type)
        if not t_base:
            raise ValueError("不支持的任务类型")

        # 2. 计算复杂度
        self.calculate_complexity(d,d_max, n,n_max, mu)
        print(f"\n{self.complexity}")

        # 3. 获取最大复杂度（示例取历史经验值）
        theta_max = 10.0  # 可根据实际情况动态调整

        # 4. 计算调整后的时间


        # 5. 选择人员级别
        self.get_level()
        cost_sum=0
        for level in self.level:
            human_value = self.levels[level]
            if self.q[level]==0:
                work_time=0
                cost_level=0
            else:
                work_time = t_base * (1 + human_value['argument'] * (self.complexity / theta_max) ** self.gamma)/(1 + human_value['argument']) *self.q[level]
                cost_level =work_time*human_value['wage']+human_value['number']*human_value['e_cost']
            cost_sum+=cost_level
            real_time=work_time/human_value['number']
            print(f"等级{level} 雇佣人数：{human_value['number']} 任务量{self.q[level]} 耗时：{real_time} 花费{cost_level}")
        # 6. 计算总成本
        print(f"总花费{cost_sum}")
        return cost_sum
    def human_time_and_cost(self, task_type, d,d_max , n, n_max,mu,e,number):
        """
                计算单个任务的人工成本
                :param n_max:
                :param d_max:
                :param task_type: 任务类型
                :param d: 数据维度
                :param n: 样本量
                :param mu: 模糊度
                :return: 人工成本（元）
                """
        # 1. 获取基础时间
        t_base = self.base_time.get(task_type)
        if not t_base:
            raise ValueError("不支持的任务类型")

        # 2. 计算复杂度
        self.calculate_complexity(d, d_max, n, n_max, mu)
        # print(f"\n{self.complexity}")

        # 3. 获取最大复杂度（示例取历史经验值）
        theta_max = 10.0  # 可根据实际情况动态调整

        # 4. 计算调整后的时间

        # 5. 选择人员级别
        self.get_level()
        human_value = self.levels[self.level]
        work_time = t_base * (1 + human_value['argument'] * (self.complexity / theta_max) ** self.gamma) / (
                    1 + human_value['argument'])
        return work_time * e * n/number ,work_time*human_value['wage']+human_value['number']*human_value['e_cost']




    def calculate_total_cost(self, tasks):
        """
        计算多任务总人工成本
        :param tasks: 任务列表[{
            'task_type': str,
            'd': int,
            'n': int,
            'mu': float
        }]
        :return: 总成本（元）
        """
        total = 0
        for task in tasks:
            try:
                print(f"\n任务{task['task_type']}:")
                cost = self.calculate_task_cost(**task)
                total += cost
            except Exception as e:
                print(f"任务计算失败: {str(e)}")
                continue
        return round(total, 2)






# # 示例用法
# if __name__ == "__main__":
#     # 初始化计算器
#     calculator = HumanCostCalculator()
#
#     # 示例任务列表
#     task_list = [
#         {
#             'task_type': 'data_labeling',
#             'd': 20,  # 数据维度
#             'd_max': 30,
#             'n': 10000,
#             'n_max': 100000000,# 样本量
#             'mu': 0.3  # 模糊度
#         },
#         {
#             'task_type': 'missing_value',
#             'd': 15,
#             'd_max': 30,
#             'n': 100000,
#             'n_max': 100000000,
#             'mu': 0.1
#         }
#     ]
#
#     # 计算总成本
#     total_cost = calculator.calculate_total_cost(task_list)
#     print(f"预测总人工成本: ¥{total_cost}")
#
#     # 查看单个任务详情
#     sample_task = task_list[1]
#     calculator.calculate_complexity(sample_task['d'],sample_task['d_max'],sample_task['n'],sample_task['n_max'],sample_task['mu'])
#     calculator.get_level(sample_task['n'])
#     print(f"\n示例任务:{sample_task['task_type']} 复杂度: {calculator.complexity:.2f}")
#     print(f"推荐人员级别: {calculator.level}")
#     calculator.calculate_task_cost(**sample_task)
#
#
#     test_components = np.array([
#         [0.3 * 20 / 30, 0.1 * math.log(1e8) / math.log(1e8), 0.6 * 0.95],  # Task 1
#         [0.3 * 25 / 30, 0.1 * math.log(1e8) / math.log(1e8), 0.6 * 0.1],  # Task 2
#         [0.3 * 30 / 30, 0.1 * math.log(1e3) / math.log(1e8), 0.6 * 0.5]  # Task 3
#     ])
#
#     CostVisualizer.plot_complexity_components( test_components)
