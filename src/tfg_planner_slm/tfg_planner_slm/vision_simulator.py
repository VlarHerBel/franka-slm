import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import random

class VisionSimulator(Node):
    def __init__(self):
        super().__init__('vision_simulator')
        # Publicamos en el topic /detected_objects
        self.publisher_ = self.create_publisher(String, '/detected_objects', 10)
        self.timer = self.create_timer(10.0, self.timer_callback) # Cambia cada 10s
        self.get_logger().info('👀 Cámara (Simulada) Iniciada')

        # Escenarios posibles (Lo que vería la cámara real)
        self.scenarios = [
            ["manzana_roja", "platano_amarillo"],
            ["tuerca_m5", "tornillo_largo", "destornillador"],
            ["cubo_azul", "esfera_roja"],
            [] # Mesa vacía
        ]

    def timer_callback(self):
        # Elegimos un escenario al azar
        current_objects = self.scenarios[0]
        # random.choice(self.scenarios)
        
        # Lo enviamos como JSON string
        msg = String()
        msg.data = json.dumps(current_objects)
        self.publisher_.publish(msg)
        
        self.get_logger().info(f'📸 La cámara ve: {current_objects}')

def main(args=None):
    rclpy.init(args=args)
    node = VisionSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()