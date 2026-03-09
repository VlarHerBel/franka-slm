import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class VisionBridge(Node):
    def __init__(self):
        super().__init__('vision_bridge')
        
        # 1. ESCUCHA A TU NODO DE PANDA_WS
        # Se suscribe al topic que ya tienes funcionando
        self.subscription = self.create_subscription(
            String,
            '/color_coordinates', 
            self.camera_callback,
            10)
            
        # 2. PUBLICA PARA EL SLM Y EXECUTOR
        self.publisher_ = self.create_publisher(String, '/detected_objects', 10)
        
        # Memoria interna
        self.detected_objects = {}
        
        # Publicamos el estado cada 0.5s para no saturar al LLM
        self.timer = self.create_timer(0.5, self.publish_state)
        
        self.get_logger().info('🌉 Vision Bridge CONECTADO. Escuchando /color_coordinates...')

    def camera_callback(self, msg):
        try:
            # Tu nodo manda: "R,0.55,-0.02,0.05"
            parts = msg.data.split(',')
            if len(parts) >= 4:
                tag = parts[0].strip()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                # TRADUCCIÓN: De "R" (Código) a "red_cube" (Semántica)
                name_map = {"R": "red_cube", "G": "green_cube", "B": "blue_cube"}
                obj_name = name_map.get(tag, tag)
                
                # Actualizamos coordenadas
                self.detected_objects[obj_name] = [x, y, z]
                
        except Exception as e:
            self.get_logger().warn(f'Error traduciendo coordenadas: {e}')

    def publish_state(self):
        if self.detected_objects:
            msg = String()
            msg.data = json.dumps(self.detected_objects)
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()