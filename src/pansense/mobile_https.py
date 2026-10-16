"""Local HTTPS certificates. Never installs a CA or changes firewall rules."""
from datetime import datetime,timedelta,timezone
from ipaddress import ip_address,ip_network
from pathlib import Path
import socket
import threading
import time


def lan_addresses():
    candidates=socket.gethostbyname_ex(socket.gethostname())[2]
    # VPN/TUN benchmark ranges (notably 198.18/15) may report is_private on
    # some Python versions; they are not the phone's Wi-Fi address.
    networks=[ip_network(v) for v in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16')]
    return list(dict.fromkeys(v for v in candidates if any(ip_address(v) in network for network in networks)))


def lan_address():
    private=lan_addresses()
    if not private: raise ValueError('没有找到局域网地址，请连接 Wi-Fi 并指定 --lan-host')
    return private[0]


def certificates(folder:Path,hostname:str):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID,NameOID
    folder.mkdir(parents=True,exist_ok=True)
    ca_key_path=folder/'ca-key.pem';ca_path=folder/'ca.pem'
    current=datetime.now(timezone.utc)
    if ca_path.is_file() and ca_key_path.is_file():
        ca=x509.load_pem_x509_certificate(ca_path.read_bytes())
        ca_key=serialization.load_pem_private_key(ca_key_path.read_bytes(),password=None)
    else:
        ca_key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'PanSense Local Camera')])
        ca=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(current-timedelta(days=1))
            .not_valid_after(current+timedelta(days=3650)).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True,key_encipherment=False,key_cert_sign=True,crl_sign=True,
                content_commitment=False,data_encipherment=False,key_agreement=False,encipher_only=False,decipher_only=False),critical=True)
            .sign(ca_key,hashes.SHA256()))
        ca_key_path.write_bytes(ca_key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
        ca_key_path.chmod(0o600);ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    server_key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    names=[x509.DNSName('localhost'),x509.IPAddress(ip_address('127.0.0.1'))]
    try: names.append(x509.IPAddress(ip_address(hostname)))
    except ValueError: names.append(x509.DNSName(hostname))
    leaf=(x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,hostname)]))
        .issuer_name(ca.subject).public_key(server_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(current-timedelta(days=1)).not_valid_after(current+timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
        .add_extension(x509.SubjectAlternativeName(names),critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),critical=False).sign(ca_key,hashes.SHA256()))
    key_path=folder/'server-key.pem';cert_path=folder/'server.pem'
    key_path.write_bytes(server_key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    key_path.chmod(0o600);cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    public_ca=folder/'PanSense-local-camera.cer';public_ca.write_bytes(ca.public_bytes(serialization.Encoding.DER))
    return {'cert':cert_path,'key':key_path,'ca':public_ca,'fingerprint':ca.fingerprint(hashes.SHA256()).hex(':').upper()}


class MobileGateway:
    def __init__(self,scans):
        self.scans=scans;self.servers=[];self.threads=[];self.lock=threading.Lock()

    def start(self,host=None,https_port=8443,setup_port=8081):
        import uvicorn
        from food3d.paths import ProjectPaths
        from .scan_api import capture_app,certificate_app
        from .scans import qr_data
        with self.lock:
            if self.servers: return
            host=host or lan_address()
            local=socket.gethostbyname_ex(socket.gethostname())[2]
            if host not in [*local,'127.0.0.1','localhost']: raise ValueError('请选择这台电脑的局域网地址')
            if https_port==setup_port: raise ValueError('HTTPS 与证书设置端口不能相同')
            # Preflight both ports before starting either service.
            probes=[]
            try:
                for port in (https_port,setup_port):
                    probe=socket.socket();probes.append(probe)
                    if hasattr(socket,'SO_EXCLUSIVEADDRUSE'): probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
                    probe.bind(('0.0.0.0',port))
            except OSError as error:
                raise ValueError('手机连接端口已被占用，请关闭重复服务或指定其他端口') from error
            finally:
                for probe in probes: probe.close()
            tls=certificates(ProjectPaths(self.scans.project).cache_root/'mobile-https',host)
            scans=self.scans
            scans.ca_path=tls['ca'];scans.certificate_fingerprint=tls['fingerprint']
            servers=[];threads=[]
            try:
                for target,port,options in [(capture_app(scans,[host,'localhost','127.0.0.1']),https_port,
                    {'ssl_keyfile':str(tls['key']),'ssl_certfile':str(tls['cert'])}),
                    (certificate_app(scans),setup_port,{})]:
                    server=uvicorn.Server(uvicorn.Config(target,host='0.0.0.0',port=port,access_log=False,**options))
                    thread=threading.Thread(target=server.run,daemon=True);thread.start()
                    servers.append(server);threads.append(thread)
                deadline=time.monotonic()+8
                while not all(server.started for server in servers):
                    if time.monotonic()>deadline or not all(thread.is_alive() for thread in threads):
                        raise ValueError('手机 HTTPS 服务未能启动，请检查端口权限和运行日志')
                    time.sleep(.05)
                scans.public_url=f'https://{host}:{https_port}'
                scans.setup_url=f'http://{host}:{setup_port}/';scans.setup_qr=qr_data(scans.setup_url)
                self.servers,self.threads=servers,threads
            except BaseException:
                for server in servers: server.should_exit=True
                for thread in threads: thread.join(timeout=3)
                raise

    def close(self):
        with self.lock:
            for server in self.servers: server.should_exit=True
            for thread in self.threads: thread.join(timeout=5)
            self.servers=[];self.threads=[];self.scans.public_url=None
