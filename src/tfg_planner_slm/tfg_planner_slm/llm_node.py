import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json

class LLMTaskPlanner(Node):
    def __init__(self):
        super().__init__('llm_task_planner')
        
        self.model = "qwen2.5:1.5b" # O el modelo que uses
        self.api_url = "http://localhost:11434/api/generate"
        
        self.current_scene_objects = [] 

        self.vision_sub = self.create_subscription(
            String,
            '/detected_objects',
            self.vision_callback,
            10)

        self.subscription = self.create_subscription(
            String,
            '/human_command',
            self.listener_callback,
            10)
        
        self.publisher_ = self.create_publisher(String, '/robot_plan', 10)
        self.get_logger().info('🚀 Nodo LLM Dinámico listo.')

    def vision_callback(self, msg):
        try:
            self.current_scene_objects = json.loads(msg.data)
            # Descomenta esto si quieres ver la memoria actualizándose sin parar:
            # self.get_logger().info(f'🧠 Memoria visual actualizada.')
        except:
            pass

    def listener_callback(self, msg):
        command = msg.data
        self.get_logger().info(f"🗣️ Nueva orden recibida: '{command}'")
        
        plan_json = self.query_ollama(command, self.current_scene_objects)
        
        if plan_json:
            # --- PROTECCIÓN: Limpiamos la respuesta por si Ollama añade markdown ---
            clean_json = plan_json.replace("```json", "").replace("```", "").strip()
            
            self.get_logger().info(f"📤 Plan limpio enviado al Executor: {clean_json}")
            msg_out = String()
            msg_out.data = clean_json
            self.publisher_.publish(msg_out)
        else:
            self.get_logger().warn("⚠️ No se pudo generar un plan (Ollama no respondió correctamente).")

    def query_ollama(self, prompt, available_objects):
        # Como available_objects ahora es un diccionario {'red_cube':[...]}, sacamos solo los nombres
        if isinstance(available_objects, dict):
            objects_str = ", ".join(available_objects.keys())
        else:
            objects_str = ", ".join(available_objects)

        system_prompt = f"""
        ERES UN CEREBRO ROBÓTICO. TU MISIÓN ES INTERPRETAR ORDENES EN LENGUAJE NATURAL.
        
        CONTEXTO ACTUAL (IDs de objetos visibles):
        [{objects_str}]
        
        INSTRUCCIONES CLAVE:
        1. El usuario te hablará en Español. Los IDs de los objetos suelen estar en Inglés (ej: 'red_cube').
        2. TU TRABAJO ES TRADUCIR Y VINCULAR: Debes entender que "cubo rojo" o "caja roja" se refiere a 'red_cube'.
        3. RAZONA: Si el usuario pide algo que se parece semánticamente a un objeto de la lista, ÚSALO.
        4. Solo si es imposible encontrar una coincidencia lógica, devuelve error.
        
        FORMATO DE RESPUESTA (JSON):
        {{"steps": [{{"action": "PICK", "target": "ID_EXACTO_DE_LA_LISTA"}}]}}
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            self.get_logger().info(f'🇪🇸 Pensando la respuesta (Contexto: {objects_str})...')
            
            # --- PROTECCIÓN: Timeout de 15 segundos para no colgar el nodo ---
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()
            
            raw_response = response.json()['response']
            self.get_logger().info(f"🤖 Respuesta cruda de Ollama: {raw_response}")
            return raw_response
            
        except requests.exceptions.Timeout:
            self.get_logger().error('❌ Ollama se ha quedado colgado pensando (Timeout).')
            return None
        except Exception as e:
            self.get_logger().error(f'❌ Error conectando con Ollama: {str(e)}')
            return None

def main(args=None):
    rclpy.init(args=args)
    node = LLMTaskPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()