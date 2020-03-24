from sys import stdout
import time

class MyPrinter():
    def __init__(self):
        self.cur_str = ""
        self.start_time = time.time()
        
    def reset_start_time(self):
        self.start_time = time.time()
        
    def print_training(self, total_samples_cnt, finished_sample_cnt, train_loss, train_acc):
        self.clear_cur_str()
        
        finished_sample_cnt = total_samples_cnt if finished_sample_cnt > total_samples_cnt else finished_sample_cnt
        
        printing_strs = [
            "Training: {}/{}".format(finished_sample_cnt, total_samples_cnt),
            "{:d} s".format(int(time.time() - self.start_time)),
            "train_loss: {:.6f}".format(train_loss),
            "train_acc: {:.6f}".format(train_acc),
        ]
        
        self.cur_str = " - ".join(printing_strs)
        stdout.write(self.cur_str)
        
    def print_test(self, test_loss, test_acc):
        print(' - Test Loss: {:.6f} - Acc: {:.6f} '.format(
            test_loss,
            test_acc,
        ))
        
    def clear_cur_str(self):
        stdout.write("\r")
#         cur_str_len = len(self.cur_str)
#         while cur_str_len > 0:
#             stdout.write("\b")
#             stdout.flush()
#             cur_str_len -= 1
    