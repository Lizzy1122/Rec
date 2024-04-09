import os


class KGBuilder:
    def __init__(self, statics_path, output_path):
        # 初始化函数，接收存放statics_train文件的路径和输出路径
        self.statics_path = statics_path
        self.output_path = output_path
        # 使用集合来存储三元组，以保证每个三元组是唯一的
        self.knowledge_graph = set()

    def add_interaction(self, student_id, question_id, correct):
        # 根据学生答题是否正确，定义关系为"掌握"或"未掌握"
        relation = "掌握" if correct == 1 else "未掌握"
        # 创建一个三元组（学生ID, 关系, 题目ID）并添加到知识图谱集合中
        self.knowledge_graph.add((student_id, relation, question_id))

    def parse_statics_train_files(self):
        # 遍历statics_path目录下所有以"statics_train"开头的文件
        for file_name in os.listdir(self.statics_path):
            if file_name.startswith("statics_train"):
                # 打开文件并读取所有行
                with open(os.path.join(self.statics_path, file_name), 'r') as file:
                    lines = file.readlines()
                # 每3行代表一个学生的数据，循环处理每个学生
                for i in range(0, len(lines), 3):
                    # 用行号除以3得到学生ID
                    student_id = i // 3
                    # 第二行是题目ID列表，将其转换为整数列表
                    questions = list(map(int, lines[i+1].strip().split(',')))
                    # 第三行是答题情况，也转换为整数列表
                    correct_answers = list(map(int, lines[i+2].strip().split(',')))
                    # 遍历题目ID和答题情况，添加到知识图谱
                    for q_id, correct in zip(questions, correct_answers):
                        self.add_interaction(student_id, q_id, correct)

    def generate_kg_final(self):
        # 生成包含所有三元组的kg_final.txt文件
        with open(os.path.join(self.output_path, "kg_final.txt"), 'w') as kg_file:
            for triple in self.knowledge_graph:
                # 将每个三元组写入文件，元素之间用制表符分隔
                kg_file.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

    def run(self):
        # 运行函数，解析文件并生成kg_final.txt
        self.parse_statics_train_files()
        self.generate_kg_final()

# 使用示例
# 创建KGBuilder实例，传入statics_train文件所在的路径和输出路径
kg_builder = KGBuilder('D:\\Developer\\Python_workspace\\KG-Policy-test\\Data\\statics', 'D:\\Developer\\Python_workspace\\KG-Policy-test\\Data\\statics')
# 运行构建知识图谱的过程
kg_builder.run()
