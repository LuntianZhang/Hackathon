// Cheap "has the view changed?" test on 32x24 grayscale thumbnails.
export const THUMB_W=32,THUMB_H=24;
export function grayThumbnail(context,source){
 context.drawImage(source,0,0,THUMB_W,THUMB_H);
 const {data}=context.getImageData(0,0,THUMB_W,THUMB_H),gray=new Uint8Array(THUMB_W*THUMB_H);
 for(let i=0;i<gray.length;i++)gray[i]=(data[i*4]*77+data[i*4+1]*150+data[i*4+2]*29)>>8;
 return gray;
}
export function meanAbsDiff(a,b){
 if(!a||!b||a.length!==b.length)return Infinity;
 let sum=0;for(let i=0;i<a.length;i++)sum+=Math.abs(a[i]-b[i]);return sum/a.length;
}
// Sensor noise on a still scene is ~1 grey level; a ~1.5 cm hand shift at 25 cm
// moves board edges by about 2 thumbnail pixels and lands well above this.
export const SAME_VIEW_DIFF=2.5;
